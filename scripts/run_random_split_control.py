"""Run an exploratory random-image-split control alongside leave-site-out studies.

This deliberately allows every site to appear in model development. It is a
negative control for distribution shift, not a transportability evaluation and
image-level unit of analysis; patient linkage is not verified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tb_certify.calibration.source_threshold import EmpiricalSensitivityThreshold
from tb_certify.data.identity_audit import patient_level_supported
from tb_certify.evaluation.metrics import evaluate_threshold
from tb_certify.evaluation.tier_a_study import TierAStudy

CALIBRATION_SIZES = (29, 46, 61, 100, 200)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _labels(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([row["label"] == "active_tb" for row in rows], dtype=int)


def _write_split_manifest(
    path: Path, rows: list[dict[str, str]], roles: dict[int, str], seed: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "site",
        "image_id",
        "patient_id",
        "label",
        "role",
        "split_unit",
        "seed",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in sorted(roles):
            row = rows[index]
            writer.writerow(
                {
                    "site": row["site"],
                    "image_id": row["image_id"],
                    "patient_id": row.get("patient_id", ""),
                    "label": row["label"],
                    "role": roles[index],
                    "split_unit": (
                        "stratified image-level; one record per image, "
                        "patient linkage not verified"
                    ),
                    "seed": str(seed),
                }
            )


def _model(head_name: str, seed: int) -> Pipeline:
    if head_name == "logistic":
        classifier: LogisticRegression | MLPClassifier = LogisticRegression(
            max_iter=2000, random_state=seed
        )
    elif head_name == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=1e-3,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported head: {head_name}")
    return Pipeline((("standardize", StandardScaler()), ("classifier", classifier)))


def run_control(
    project_root: Path,
    embedding_dir: Path,
    output_dir: Path,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    metadata = json.loads((embedding_dir / "metadata.json").read_text(encoding="utf-8"))
    features = np.load(embedding_dir / "embeddings.npy", mmap_mode="r")
    rows = _read_csv(embedding_dir / "manifest.csv")
    labels = _labels(rows)
    if features.shape[0] != len(rows) or set(labels) != {0, 1}:
        raise ValueError("Embeddings and binary manifest labels must align.")

    audit_path = project_root / "artifacts" / "audits" / "image_identity.json"
    identity_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    duplicate_aliases = set()
    for site, site_audit in identity_audit[
        "exact_decoded_rgb_pixel_duplicates"
    ].items():
        for group in site_audit["duplicate_groups"]:
            duplicate_aliases.update((site, image_id) for image_id in group[1:])
    all_indices = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if (row["site"], row["image_id"]) not in duplicate_aliases
        ],
        dtype=int,
    )
    development_indices, evaluation_indices = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=seed,
        stratify=labels[all_indices],
    )
    model_indices, calibration_indices = train_test_split(
        development_indices,
        test_size=0.25,
        random_state=seed,
        stratify=labels[development_indices],
    )
    model_train_indices, source_validation_indices = train_test_split(
        model_indices,
        test_size=0.20,
        random_state=seed,
        stratify=labels[model_indices],
    )
    calibration_positive_indices = calibration_indices[labels[calibration_indices] == 1]
    if calibration_positive_indices.size < max(CALIBRATION_SIZES):
        raise ValueError("Random calibration split lacks positives for the study grid.")

    roles = {
        **{int(index): "model_train" for index in model_train_indices},
        **{int(index): "source_validation" for index in source_validation_indices},
        **{int(index): "calibration" for index in calibration_indices},
        **{int(index): "evaluation" for index in evaluation_indices},
    }
    roles.update(
        {
            index: "exact_pixel_duplicate_excluded"
            for index, row in enumerate(rows)
            if (row["site"], row["image_id"]) in duplicate_aliases
        }
    )
    split_path = (
        project_root / "data" / "splits" / f"random_image_control_seed{seed}.csv"
    )
    _write_split_manifest(split_path, rows, roles, seed)

    results: dict[str, Any] = {
        "study": "random_image_split_negative_control exploratory evaluation",
        "purpose": (
            "Contrast an image-randomized, site-mixed split with leave-site-out; "
            "not a transportability study."
        ),
        "split_unit": "image-level; patient linkage not verified",
        "patient_level_claim": patient_level_supported(project_root),
        "pac_claim": False,
        "seed": seed,
        "repetitions": repetitions,
        "split_manifest": str(split_path),
        "counts": {
            "model_train": int(model_train_indices.size),
            "source_validation": int(source_validation_indices.size),
            "calibration": int(calibration_indices.size),
            "calibration_positive": int(calibration_positive_indices.size),
            "evaluation": int(evaluation_indices.size),
            "evaluation_positive": int(np.sum(labels[evaluation_indices] == 1)),
            "evaluation_negative": int(np.sum(labels[evaluation_indices] == 0)),
            "exact_duplicate_aliases_excluded": len(duplicate_aliases),
        },
        "encoder": metadata,
        "heads": {},
    }

    for head_name in ("logistic", "mlp"):
        model = _model(head_name, seed)
        model.fit(features[model_train_indices], labels[model_train_indices])
        validation_scores = model.predict_proba(features[source_validation_indices])[
            :, 1
        ]
        source_threshold = (
            EmpiricalSensitivityThreshold(0.90)
            .fit(validation_scores[labels[source_validation_indices] == 1])
            .threshold
        )
        evaluation_scores = model.predict_proba(features[evaluation_indices])[:, 1]
        calibration_scores = model.predict_proba(
            features[calibration_positive_indices]
        )[:, 1]
        tier_a = TierAStudy(target_specificity=0.70, confidence=0.95).run(
            calibration_positive_scores=calibration_scores,
            evaluation_scores=evaluation_scores,
            evaluation_labels=labels[evaluation_indices],
            calibration_sizes=CALIBRATION_SIZES,
            repetitions=repetitions,
            seed=seed,
            split_unit=(
                "stratified image-level; site-mixed; patient linkage not verified"
            ),
        )
        results["heads"][head_name] = {
            "source_threshold_target_sensitivity": 0.90,
            "source_validation_threshold": float(source_threshold),
            "source_validation_metrics": evaluate_threshold(
                validation_scores, labels[source_validation_indices], source_threshold
            ).__dict__,
            "evaluation_b0_metrics": evaluate_threshold(
                evaluation_scores, labels[evaluation_indices], source_threshold
            ).__dict__,
            "evaluation_b0_metrics_by_site": {
                site: evaluate_threshold(
                    evaluation_scores[
                        np.asarray(
                            [
                                rows[index]["site"] == site
                                for index in evaluation_indices
                            ]
                        )
                    ],
                    labels[evaluation_indices][
                        np.asarray(
                            [
                                rows[index]["site"] == site
                                for index in evaluation_indices
                            ]
                        )
                    ],
                    source_threshold,
                ).__dict__
                for site in sorted(
                    {rows[index]["site"] for index in evaluation_indices}
                )
            },
            "tier_a_exploratory": tier_a.to_dict(),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if arguments.repetitions < 1:
        parser.error("--repetitions must be positive")
    results = run_control(
        project_root=arguments.project_root.resolve(),
        embedding_dir=arguments.embedding_dir.resolve(),
        output_dir=arguments.output_dir,
        repetitions=arguments.repetitions,
        seed=arguments.seed,
    )
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
