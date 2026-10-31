"""Probe whether frozen embeddings expose a dataset/site shortcut.

This is a diagnostic only: high site-prediction accuracy establishes detectable
domain information in the embeddings, not a causal explanation for TB scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def run_probe(
    embedding_dir: Path,
    seed: int = 42,
    folds: int = 5,
    identity_audit_path: Path | None = None,
) -> dict[str, Any]:
    embedding_dir = embedding_dir.resolve()
    features = np.load(embedding_dir / "embeddings.npy", mmap_mode="r")
    rows = _read_rows(embedding_dir / "manifest.csv")
    if features.ndim != 2 or features.shape[0] != len(rows):
        raise ValueError("Embedding rows must align one-to-one with manifest rows.")
    alias_keys: set[tuple[str, str]] = set()
    if identity_audit_path is not None:
        identity_audit = json.loads(identity_audit_path.read_text(encoding="utf-8"))
        alias_keys = {
            (site, image_id)
            for site, site_audit in identity_audit[
                "exact_decoded_rgb_pixel_duplicates"
            ].items()
            for group in site_audit["duplicate_groups"]
            for image_id in group[1:]
        }
    keep = np.asarray(
        [(row["site"], row.get("image_id", "")) not in alias_keys for row in rows],
        dtype=bool,
    )
    rows = [row for row, include in zip(rows, keep, strict=True) if include]
    features = features[keep]
    sites = np.asarray([row["site"] for row in rows])
    if len(set(sites)) < 2:
        raise ValueError("At least two sites are required for a site-origin probe.")
    site_counts = {site: int(np.sum(sites == site)) for site in sorted(set(sites))}
    if min(site_counts.values()) < folds:
        raise ValueError("Every site needs at least as many records as folds.")

    encoder = LabelEncoder()
    labels = encoder.fit_transform(sites)
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(
        model, features, labels, cv=splitter, method="predict_proba"
    )
    predictions = np.argmax(probabilities, axis=1)
    if len(encoder.classes_) == 2:
        macro_auc = float(roc_auc_score(labels, probabilities[:, 1]))
    else:
        macro_auc = float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        )
    return {
        "purpose": "exploratory site-origin shortcut probe",
        "interpretation_limit": (
            "A high score means the frozen embedding retains site information; it "
            "does not identify its cause or prove TB-label shortcutting."
        ),
        "split_unit": "image-level; patient linkage not verified",
        "exact_pixel_duplicate_aliases_excluded": int(np.sum(~keep)),
        "identity_audit_path": (
            str(identity_audit_path.resolve())
            if identity_audit_path is not None
            else None
        ),
        "seed": seed,
        "folds": folds,
        "embedding_dir": str(embedding_dir),
        "site_counts": site_counts,
        "classes": encoder.classes_.tolist(),
        "cross_validated_accuracy": float(accuracy_score(labels, predictions)),
        "cross_validated_balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "cross_validated_macro_ovr_auc": macro_auc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--identity-audit",
        type=Path,
        default=Path("artifacts/audits/image_identity.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.folds < 2:
        parser.error("--folds must be at least two")
    identity_audit = arguments.identity_audit
    if not identity_audit.is_absolute():
        identity_audit = Path.cwd() / identity_audit
    result = run_probe(
        arguments.embedding_dir,
        arguments.seed,
        arguments.folds,
        identity_audit if identity_audit.is_file() else None,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
