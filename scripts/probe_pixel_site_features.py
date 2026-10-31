"""Probe site origin using simple image-intensity and native-size features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _source_bytes(
    row: dict[str, str], raw_dir: Path, archive: zipfile.ZipFile
) -> bytes:
    if row["site"] == "TBX11K":
        return archive.read(row["source_member"])
    return (raw_dir / row["source_member"]).read_bytes()


def _pixel_features(payload: bytes) -> list[float]:
    with Image.open(BytesIO(payload)) as image:
        width, height = image.size
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    quantiles = np.percentile(gray, (1, 5, 50, 95, 99))
    normalized = gray / 255.0
    return [
        np.log1p(width),
        np.log1p(height),
        np.log1p(width * height),
        width / height,
        float(np.mean(normalized)),
        float(np.std(normalized)),
        *(float(value / 255.0) for value in quantiles),
        float(np.mean(gray <= 16)),
        float(np.mean(gray >= 239)),
        float((quantiles[-1] - quantiles[0]) / 255.0),
    ]


def run_probe(
    project_root: Path, manifest_path: Path, seed: int = 42, folds: int = 5
) -> dict[str, Any]:
    project_root = project_root.resolve()
    rows = _read_rows(manifest_path)
    audit_path = project_root / "artifacts" / "audits" / "image_identity.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    aliases: set[tuple[str, str]] = set()
    for site, site_audit in audit["exact_decoded_rgb_pixel_duplicates"].items():
        for group in site_audit["duplicate_groups"]:
            aliases.update((site, image_id) for image_id in group[1:])
    rows = [row for row in rows if (row["site"], row["image_id"]) not in aliases]
    raw_dir = project_root / "data" / "raw"
    feature_rows: list[list[float]] = []
    with zipfile.ZipFile(raw_dir / "TBX11K.zip") as archive:
        for index, row in enumerate(rows, start=1):
            feature_rows.append(_pixel_features(_source_bytes(row, raw_dir, archive)))
            if index % 1000 == 0:
                print(f"Read {index}/{len(rows)} images", flush=True)

    features = np.asarray(feature_rows, dtype=float)
    sites = np.asarray([row["site"] for row in rows])
    encoder = LabelEncoder()
    labels = encoder.fit_transform(sites)
    if np.bincount(labels).min() < folds:
        raise ValueError("Every site needs at least as many records as folds.")
    model = Pipeline(
        (
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=seed
                ),
            ),
        )
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(
        model, features, labels, cv=splitter, method="predict_proba"
    )
    predictions = np.argmax(probabilities, axis=1)
    auc = float(
        roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
    )
    with manifest_path.open("rb") as stream:
        manifest_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "purpose": "diagnostic comparison for the embedding site-origin probe",
        "feature_groups": [
            "native width, height, and area",
            "aspect ratio",
            "grayscale mean, standard deviation, quantiles, and tail fractions",
        ],
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": manifest_sha256,
        "split_unit": "image-level; patient linkage not verified",
        "seed": seed,
        "folds": folds,
        "exact_pixel_duplicate_aliases_excluded": len(aliases),
        "image_count": len(rows),
        "site_counts": {
            site: int(np.sum(sites == site)) for site in sorted(set(sites))
        },
        "classes": encoder.classes_.tolist(),
        "cross_validated_accuracy": float(accuracy_score(labels, predictions)),
        "cross_validated_balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "cross_validated_macro_ovr_auc": auc,
        "interpretation_limit": (
            "The probe measures site information in a small, hand-designed feature "
            "set. It neither establishes causal shortcut use nor explains the "
            "encoder probe's performance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/primary/manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/runs/primary_tbx11k_loso/pixel_site_probe.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    manifest = arguments.manifest
    if not manifest.is_absolute():
        manifest = root / manifest
    output = arguments.output
    if not output.is_absolute():
        output = root / output
    result = run_probe(root, manifest, arguments.seed, arguments.folds)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
