"""Run an exploratory leave-site-out encoder/head study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tb_certify.calibration.risk_control import (
    ImportanceWeightedLTTCalibrator,
    MarginalCRCCalibrator,
    RiskControlStatus,
)
from tb_certify.calibration.source_threshold import EmpiricalSensitivityThreshold
from tb_certify.data.identity_audit import patient_level_supported
from tb_certify.evaluation.metrics import evaluate_threshold
from tb_certify.evaluation.selective_study import SelectiveVerificationStudy
from tb_certify.evaluation.tier_a_study import TierAStudy
from tb_certify.training.source_head import SourceOnlyLogisticHead

PRIMARY_SOURCE_SITES = ("Montgomery", "Shenzhen")
SECONDARY_SOURCE_SITES = ("Montgomery", "TBX11K")
CALIBRATION_SIZES = (29, 46, 61, 100, 200)
SEED = 42


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _binary_label(label: str) -> int:
    return int(label == "active_tb")


def _source_split(
    rows: list[dict[str, str]], source_sites: tuple[str, ...], seed: int
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    train_indices: list[int] = []
    validation_indices: list[int] = []
    split_manifest: list[dict[str, str]] = []
    for site in source_sites:
        if site == "TBX11K":
            train_indices_for_site = [
                index
                for index, row in enumerate(rows)
                if row["site"] == site and row["source_split"] == "train"
            ]
            validation_indices_for_site = [
                index
                for index, row in enumerate(rows)
                if row["site"] == site and row["source_split"] == "validation"
            ]
            train = np.asarray(train_indices_for_site, dtype=int)
            validation = np.asarray(validation_indices_for_site, dtype=int)
            site_indices = np.concatenate((train, validation))
        else:
            site_indices = np.asarray(
                [index for index, row in enumerate(rows) if row["site"] == site],
                dtype=int,
            )
        if site_indices.size == 0:
            raise ValueError(f"No rows found for source site {site}.")
        if site != "TBX11K":
            labels = np.asarray(
                [_binary_label(rows[index]["label"]) for index in site_indices]
            )
            train, validation = train_test_split(
                site_indices,
                test_size=0.20,
                random_state=seed,
                stratify=labels,
            )
        train_indices.extend(int(index) for index in train)
        validation_indices.extend(int(index) for index in validation)
        train_set = set(int(index) for index in train)
        for index in site_indices:
            row = rows[int(index)]
            split_manifest.append(
                {
                    "site": site,
                    "image_id": row["image_id"],
                    "patient_id": row.get("patient_id", ""),
                    "source_split": row["source_split"],
                    "label": row["label"],
                    "role": "train" if int(index) in train_set else "validation",
                    "split_unit": "image-level; patient linkage not verified",
                    "seed": str(seed),
                }
            )
    return (
        np.asarray(train_indices, dtype=int),
        np.asarray(validation_indices, dtype=int),
        split_manifest,
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fit_head(
    head_name: str,
    features: np.ndarray,
    rows: list[dict[str, str]],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    target_site: str,
    source_sites: tuple[str, ...],
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray, float]:
    train_labels = np.asarray(
        [_binary_label(rows[index]["label"]) for index in train_indices], dtype=int
    )
    validation_labels = np.asarray(
        [_binary_label(rows[index]["label"]) for index in validation_indices], dtype=int
    )
    train_sites = [rows[index]["site"] for index in train_indices]
    validation_sites = [rows[index]["site"] for index in validation_indices]

    if head_name == "logistic":
        head = SourceOnlyLogisticHead(seed=seed).fit(
            features[train_indices],
            train_labels,
            train_sites,
            source_sites=source_sites,
            held_out_target_site=target_site,
        )
        validation_scores = head.score(features[validation_indices])
        threshold = head.choose_source_threshold(
            features[validation_indices],
            validation_labels,
            validation_sites,
            target_sensitivity=0.90,
        ).threshold
        return head, validation_scores, validation_labels, threshold

    head = Pipeline(
        steps=[
            ("standardize", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128,),
                    alpha=1e-3,
                    max_iter=300,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=seed,
                ),
            ),
        ]
    )
    head.fit(features[train_indices], train_labels)
    validation_scores = head.predict_proba(features[validation_indices])[:, 1]
    threshold = (
        EmpiricalSensitivityThreshold(0.90)
        .fit(validation_scores[validation_labels == 1])
        .threshold
    )
    return head, validation_scores, validation_labels, threshold


def _scores(head: Any, head_name: str, features: np.ndarray) -> np.ndarray:
    if head_name == "logistic":
        return np.asarray(head.score(features), dtype=float)
    return np.asarray(head.predict_proba(features)[:, 1], dtype=float)


def _risk_control_comparison(
    calibration_scores: np.ndarray,
    evaluation_scores: np.ndarray,
    evaluation_labels: np.ndarray,
    source_validation_scores: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Compare CRC-family rules under the explicit unit-weight baseline."""
    unit_weights = np.ones(calibration_scores.size, dtype=float)
    crc = MarginalCRCCalibrator(0.10).calibrate(calibration_scores)
    ltt_thresholds = np.unique(
        np.quantile(source_validation_scores, np.linspace(0.01, 0.99, 25))
    )
    if not ltt_thresholds.size:
        raise ValueError("Source validation scores cannot define an IW-LTT grid.")
    ltt = ImportanceWeightedLTTCalibrator(
        target_fnr=0.10, failure_probability=0.05, seed=seed
    ).calibrate(
        calibration_scores,
        unit_weights,
        weight_upper_bound=1.0,
        thresholds=ltt_thresholds,
    )
    methods: dict[str, Any] = {}
    for name, certificate in (("crc", crc), ("iw_ltt_unit_weights", ltt)):
        if certificate is None:
            methods[name] = {"status": "not_evaluated", "reason": "empty_evaluation"}
            continue
        record = {
            key: value.value if isinstance(value, RiskControlStatus) else value
            for key, value in certificate.__dict__.items()
        }
        if certificate.status is RiskControlStatus.ACCEPTED:
            record["evaluation_metrics"] = evaluate_threshold(
                evaluation_scores, evaluation_labels, certificate.threshold
            ).__dict__
        methods[name] = record
    return {
        "weight_assumption": "unit density ratios; exchangeable positive examples",
        "shift_adjusted": False,
        "unit_weight_w_crc_note": (
            "W-CRC with calibration and test weights equal to one is algebraically "
            "identical to CRC and is not reported as a separate method."
        ),
        "ltt_threshold_grid": "25 source-validation score quantiles from 0.01 to 0.99",
        "ltt_thresholds": [float(value) for value in ltt_thresholds],
        "methods": methods,
    }


def run_study(
    project_root: Path,
    embedding_dir: Path,
    output_dir: Path | None,
    repetitions: int,
    selective_repetitions: int,
    seed: int,
    target_site: str,
) -> dict[str, Any]:
    embedding_dir = embedding_dir.resolve()
    metadata = json.loads((embedding_dir / "metadata.json").read_text(encoding="utf-8"))
    features = np.load(embedding_dir / "embeddings.npy", mmap_mode="r")
    rows = _read_csv(embedding_dir / "manifest.csv")
    if features.ndim != 2 or features.shape[0] != len(rows):
        raise ValueError("Embedding rows must align one-to-one with the manifest.")
    if not np.all(np.isfinite(features)):
        raise ValueError("Embedding matrix contains non-finite values.")
    keys = [(row["site"], row["image_id"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Manifest has duplicate site/image_id pairs.")
    if set(row["site"] for row in rows).difference(
        {"Montgomery", "Shenzhen", "TBX11K"}
    ):
        raise ValueError("Manifest contains an unrecognized site.")

    if target_site == "TBX11K":
        source_sites = PRIMARY_SOURCE_SITES
        protocol_name = "primary_tbx11k_loso"
    elif target_site == "Shenzhen":
        source_sites = SECONDARY_SOURCE_SITES
        protocol_name = "secondary_shenzhen_loso"
    else:
        raise ValueError(f"Unsupported target site: {target_site}")
    train_indices, validation_indices, split_manifest = _source_split(
        rows, source_sites, seed
    )
    target_site_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["site"] == target_site],
        dtype=int,
    )
    if target_site == "TBX11K":
        target_train_indices = np.asarray(
            [
                index
                for index in target_site_indices
                if rows[index]["source_split"] == "train"
            ],
            dtype=int,
        )
        target_evaluation_indices = np.asarray(
            [
                index
                for index in target_site_indices
                if rows[index]["source_split"] == "validation"
            ],
            dtype=int,
        )
        target_split_unit = (
            "official TBX11K train/validation split; patient linkage not verified"  # noqa: E501
        )
    else:
        target_split_labels = np.asarray(
            [_binary_label(rows[index]["label"]) for index in target_site_indices]
        )
        target_train_indices, target_evaluation_indices = train_test_split(
            target_site_indices,
            test_size=0.50,
            random_state=seed,
            stratify=target_split_labels,
        )
        target_split_unit = "stratified 50/50 image split; patient linkage not verified"
    calibration_indices = np.asarray(
        [
            index
            for index in target_train_indices
            if rows[index]["label"] == "active_tb"
        ],
        dtype=int,
    )
    if not calibration_indices.size or not target_evaluation_indices.size:
        raise ValueError("Target calibration and evaluation cohorts are required.")

    for index in target_train_indices:
        row = rows[int(index)]
        split_manifest.append(
            {
                "site": target_site,
                "image_id": row["image_id"],
                "patient_id": row.get("patient_id", ""),
                "source_split": row["source_split"],
                "label": row["label"],
                "role": (
                    "calibration_positive"
                    if row["label"] == "active_tb"
                    else "calibration_split_negative_unused"
                ),
                "split_unit": target_split_unit,
                "seed": str(seed),
            }
        )
    for index in target_evaluation_indices:
        row = rows[int(index)]
        split_manifest.append(
            {
                "site": target_site,
                "image_id": row["image_id"],
                "patient_id": row.get("patient_id", ""),
                "source_split": row["source_split"],
                "label": row["label"],
                "role": "evaluation",
                "split_unit": target_split_unit,
                "seed": str(seed),
            }
        )
    split_path = (
        project_root / "data" / "splits" / f"{protocol_name}_image_split_seed{seed}.csv"
    )
    _write_csv(split_path, split_manifest)
    calibration_sizes = list(CALIBRATION_SIZES)
    if output_dir is None:
        revision = metadata.get("revision", "unknown")[:12]
        output_dir = (
            project_root
            / "experiments"
            / "runs"
            / protocol_name
            / f"{metadata['encoder_key']}-{revision}"
        )
    elif not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "study": f"{protocol_name} exploratory image-level evaluation (patient linkage not verified)",  # noqa: E501
        "target_site": target_site,
        "source_sites": list(source_sites),
        "split_unit": "image-level; patient linkage not verified",
        "patient_level_claim": patient_level_supported(project_root),
        "pac_claim": False,
        "interpretation_limit": (
            "Tier A outputs are exploratory image-level estimates; patient linkage, representativeness, "  # noqa: E501
            "i.i.d. sampling, label accuracy, and stationarity remain assumptions."
        ),
        "seed": seed,
        "repetitions": repetitions,
        "source_train_count": int(len(train_indices)),
        "source_validation_count": int(len(validation_indices)),
        "target_calibration_split_count": int(len(target_train_indices)),
        "target_calibration_pool_count": int(len(calibration_indices)),
        "target_calibration_positive_count": int(len(calibration_indices)),
        "target_evaluation_count": int(len(target_evaluation_indices)),
        "target_evaluation_positive_count": int(
            sum(
                rows[index]["label"] == "active_tb"
                for index in target_evaluation_indices
            )
        ),
        "target_evaluation_negative_count": int(
            sum(
                rows[index]["label"] != "active_tb"
                for index in target_evaluation_indices
            )
        ),
        "split_manifest": str(split_path),
        "encoder": metadata,
        "heads": {},
    }

    target_labels = np.asarray(
        [_binary_label(rows[index]["label"]) for index in target_evaluation_indices],
        dtype=int,
    )
    for head_name in ("logistic", "mlp"):
        (
            head,
            source_validation_scores,
            source_validation_labels,
            source_threshold,
        ) = _fit_head(
            head_name,
            features,
            rows,
            train_indices,
            validation_indices,
            target_site,
            source_sites,
            seed,
        )
        source_metrics = evaluate_threshold(
            source_validation_scores, source_validation_labels, source_threshold
        )
        target_scores = _scores(head, head_name, features[target_evaluation_indices])
        target_b0_metrics = evaluate_threshold(
            target_scores, target_labels, source_threshold
        )
        target_calibration_scores = _scores(
            head, head_name, features[calibration_indices]
        )
        target_calibration_all_scores = _scores(
            head, head_name, features[target_train_indices]
        )
        target_calibration_all_labels = np.asarray(
            [_binary_label(rows[index]["label"]) for index in target_train_indices],
            dtype=int,
        )
        risk_control_comparison = _risk_control_comparison(
            target_calibration_scores,
            target_scores,
            target_labels,
            source_validation_scores,
            seed,
        )
        study = TierAStudy(target_specificity=0.70, confidence=0.95).run(
            calibration_positive_scores=target_calibration_scores,
            evaluation_scores=target_scores,
            evaluation_labels=target_labels,
            calibration_sizes=calibration_sizes,
            repetitions=repetitions,
            seed=seed,
            split_unit=f"{target_split_unit}; patient linkage not verified",
        )
        negative_control_masks = (
            {
                "healthy_only": np.asarray(
                    [
                        rows[index]["label"] in ("active_tb", "healthy")
                        for index in target_evaluation_indices
                    ],
                    dtype=bool,
                ),
                "healthy_plus_non_tb": np.ones(target_labels.size, dtype=bool),
            }
            if target_site == "TBX11K"
            else {}
        )
        negative_controls: dict[str, Any] = {}
        for control_name, control_mask in negative_control_masks.items():
            control_study = TierAStudy(target_specificity=0.70, confidence=0.95).run(
                calibration_positive_scores=target_calibration_scores,
                evaluation_scores=target_scores[control_mask],
                evaluation_labels=target_labels[control_mask],
                calibration_sizes=calibration_sizes,
                repetitions=repetitions,
                seed=seed,
                split_unit=f"{target_split_unit}; patient linkage not verified",
            )
            negative_controls[control_name] = {
                "evaluation_count": int(np.sum(control_mask)),
                "evaluation_positive_count": int(
                    np.sum(target_labels[control_mask] == 1)
                ),
                "evaluation_negative_count": int(
                    np.sum(target_labels[control_mask] == 0)
                ),
                "b0_metrics": evaluate_threshold(
                    target_scores[control_mask],
                    target_labels[control_mask],
                    source_threshold,
                ).__dict__,
                "tier_a_exploratory": control_study.to_dict(),
            }
        selective_verification: dict[str, Any] | None = None
        if selective_repetitions:
            fixed_grid = np.unique(
                np.quantile(source_validation_scores, np.linspace(0.01, 0.99, 25))
            )
            if fixed_grid.size < 2:
                raise ValueError(
                    "Source score distribution cannot define a threshold grid."
                )
            selective_verification = (
                SelectiveVerificationStudy()
                .run(
                    calibration_scores=target_calibration_all_scores,
                    calibration_labels=target_calibration_all_labels,
                    evaluation_scores=target_scores,
                    evaluation_labels=target_labels,
                    thresholds=tuple(float(value) for value in fixed_grid),
                    reference_threshold=source_threshold,
                    low_score_probabilities=(0.10, 0.25, 0.50, 1.00),
                    repetitions=selective_repetitions,
                    seed=seed,
                    split_unit=(
                        f"{target_split_unit}; patient linkage not verified; "
                        "fully labelled target split used only to simulate verification"
                    ),
                )
                .to_dict()
            )
        results["heads"][head_name] = {
            "source_threshold_target_sensitivity": 0.90,
            "source_validation_threshold": float(source_threshold),
            "source_validation_metrics": source_metrics.__dict__,
            "target_b0_metrics": target_b0_metrics.__dict__,
            "tier_a_exploratory": study.to_dict(),
            "risk_control_comparison": risk_control_comparison,
            "negative_controls": negative_controls,
            "selective_verification_exploratory": selective_verification,
        }

    result_path = output_dir / "results.json"
    result_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    results["result_path"] = str(result_path)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--target-site", choices=("TBX11K", "Shenzhen"), default="TBX11K"
    )
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument(
        "--selective-repetitions",
        type=int,
        default=0,
        help="selective-verification simulations per policy (default: disabled)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()
    if arguments.repetitions < 1:
        parser.error("--repetitions must be positive")
    if arguments.selective_repetitions < 0:
        parser.error("--selective-repetitions must not be negative")
    result = run_study(
        arguments.project_root.resolve(),
        arguments.embedding_dir,
        arguments.output_dir,
        arguments.repetitions,
        arguments.selective_repetitions,
        arguments.seed,
        arguments.target_site,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
