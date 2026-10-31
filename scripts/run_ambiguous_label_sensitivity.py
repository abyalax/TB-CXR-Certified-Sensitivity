"""Audit TBX11K results under two mappings for obsolete/unresolved labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scripts.run_primary_real_study import (
    PRIMARY_SOURCE_SITES,
    _fit_head,
    _read_csv,
    _scores,
    _source_split,
)

from tb_certify.data.identity_audit import patient_level_supported
from tb_certify.data.preprocessing import CXRImagePreprocessor, CXRPreprocessingConfig
from tb_certify.data.records import ClinicalClass, ImageRecord
from tb_certify.data.tbx11k import TBX11KArchiveCatalog
from tb_certify.evaluation.metrics import evaluate_threshold
from tb_certify.evaluation.tier_a_study import TierAStudy
from tb_certify.models.embeddings import extract_from_project

AMBIGUOUS_LABELS = {ClinicalClass.OBSOLETE_TB, ClinicalClass.UNRESOLVED_TB}
CALIBRATION_SIZES = (29, 46, 61, 100, 200)


def _cache_ambiguous_images(
    project_root: Path,
    encoder_key: str,
    records: tuple[ImageRecord, ...],
) -> Path:
    image_size = 518 if encoder_key == "chest_xray" else 224
    cache_root = project_root / "data" / "processed" / "label_sensitivity" / encoder_key
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "manifest.csv"
    if manifest_path.exists():
        return cache_root

    preprocessor = CXRImagePreprocessor(CXRPreprocessingConfig(image_size=image_size))
    manifest_rows: list[dict[str, Any]] = []
    catalog = TBX11KArchiveCatalog(project_root / "data" / "raw" / "TBX11K.zip")
    with catalog.open_image_reader() as reader:
        for record in records:
            if record.label is None or record.source_member is None:
                raise ValueError("Ambiguous records need labels and archive members.")
            identity = f"{record.site}/{record.source_member}".encode()
            cache_name = f"{hashlib.sha256(identity).hexdigest()[:24]}.png"
            relative_cache_path = Path("images") / cache_name
            cache_path = cache_root / relative_cache_path
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            image_array = preprocessor.process_bytes(reader.read_image_bytes(record))
            payload_stream = BytesIO()
            Image.fromarray(image_array).save(
                payload_stream, format="PNG", optimize=False
            )
            payload = payload_stream.getvalue()
            cache_path.write_bytes(payload)
            manifest_rows.append(
                {
                    "site": record.site,
                    "image_id": record.image_id,
                    "patient_id": "",
                    "label": record.label.value,
                    "source_split": record.source_split or "",
                    "source_member": record.source_member,
                    "cache_path": relative_cache_path.as_posix(),
                    "height": image_array.shape[0],
                    "width": image_array.shape[1],
                    "channels": image_array.shape[2],
                    "cache_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )

    fields = tuple(manifest_rows[0])
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    metadata = {
        "preprocessing": asdict(CXRPreprocessingConfig(image_size=image_size)),
        "pixel_format": "uint8 RGB PNG",
        "crop_policy": "full_image_no_crop",
        "label_policy": (
            "obsolete and unresolved TBX11K records retained for sensitivity analysis"
        ),
        "patient_id_policy": (
            "patient_id remains blank; image_id is not treated as patient ID"
        ),
        "record_count": len(records),
        "successful_images": len(records),
        "decode_failures": 0,
        "random_seed": "not_applicable_deterministic_preprocessing",
    }
    (cache_root / "preprocessing.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return cache_root


def run_analysis(
    project_root: Path,
    embedding_dir: Path,
    encoder_key: str,
    output_dir: Path,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    embedding_dir = embedding_dir.resolve()
    output_dir = output_dir.resolve()
    primary_rows = _read_csv(embedding_dir / "manifest.csv")
    primary_features = np.load(embedding_dir / "embeddings.npy", mmap_mode="r")
    if primary_features.ndim != 2 or primary_features.shape[0] != len(primary_rows):
        raise ValueError(
            "Primary embeddings must align one-to-one with their manifest."
        )

    catalog = TBX11KArchiveCatalog(project_root / "data" / "raw" / "TBX11K.zip")
    ambiguous_records = tuple(
        record
        for record in catalog.public_records("trainval")
        if record.label in AMBIGUOUS_LABELS
    )
    cache_root = _cache_ambiguous_images(project_root, encoder_key, ambiguous_records)
    ambiguous_embedding_dir = (
        project_root / "artifacts" / "embeddings" / "label_sensitivity" / encoder_key
    )
    ambiguous_metadata_path = ambiguous_embedding_dir / "metadata.json"
    if ambiguous_metadata_path.is_file():
        ambiguous_metadata = json.loads(
            ambiguous_metadata_path.read_text(encoding="utf-8")
        )
    else:
        ambiguous_metadata = extract_from_project(
            project_root=project_root,
            encoder_key=encoder_key,
            batch_size=None,
            device_name="auto",
            output_dir=ambiguous_embedding_dir,
            cache_root=cache_root,
        )
    ambiguous_rows = _read_csv(ambiguous_embedding_dir / "manifest.csv")
    ambiguous_features = np.load(
        ambiguous_embedding_dir / "embeddings.npy", mmap_mode="r"
    )
    ambiguous_score_lookup: dict[tuple[str, str], tuple[str, float]] = {}
    identity_audit = json.loads(
        (project_root / "artifacts" / "audits" / "image_identity.json").read_text(
            encoding="utf-8"
        )
    )
    duplicate_aliases = {
        (site, image_id)
        for site, site_audit in identity_audit[
            "exact_decoded_rgb_pixel_duplicates"
        ].items()
        for group in site_audit["duplicate_groups"]
        for image_id in group[1:]
    }

    source_train, source_validation, _ = _source_split(
        primary_rows, PRIMARY_SOURCE_SITES, seed
    )
    scenario_results: dict[str, Any] = {}
    for head_name in ("logistic", "mlp"):
        head, _, _, source_threshold = _fit_head(
            head_name,
            primary_features,
            primary_rows,
            source_train,
            source_validation,
            "TBX11K",
            PRIMARY_SOURCE_SITES,
            seed,
        )
        ambiguous_scores = _scores(head, head_name, ambiguous_features)
        for row, score in zip(ambiguous_rows, ambiguous_scores, strict=True):
            ambiguous_score_lookup[(row["site"], row["image_id"])] = (
                row["source_split"],
                float(score),
            )

        target_indices = [
            index for index, row in enumerate(primary_rows) if row["site"] == "TBX11K"
        ]
        target_scores = _scores(head, head_name, primary_features[target_indices])
        target_rows = [primary_rows[index] for index in target_indices]
        train_active_scores = [
            float(score)
            for row, score in zip(target_rows, target_scores, strict=True)
            if row["source_split"] == "train" and row["label"] == "active_tb"
        ]
        validation_rows: list[dict[str, str]] = []
        validation_scores: list[float] = []
        for row, score in zip(target_rows, target_scores, strict=True):
            if (
                row["source_split"] == "validation"
                and (
                    row["site"],
                    row["image_id"],
                )
                not in duplicate_aliases
            ):
                validation_rows.append(row)
                validation_scores.append(float(score))
        for row in ambiguous_rows:
            if row["source_split"] == "validation":
                validation_rows.append(row)
                validation_scores.append(
                    ambiguous_score_lookup[(row["site"], row["image_id"])][1]
                )

        per_head: dict[str, Any] = {
            "source_threshold": float(source_threshold),
            "scenarios": {},
        }
        for scenario, ambiguous_label in (
            ("all_ambiguous_positive", 1),
            ("all_ambiguous_negative", 0),
        ):
            calibration_positive_scores = list(train_active_scores)
            for row in ambiguous_rows:
                if row["source_split"] == "train" and ambiguous_label == 1:
                    calibration_positive_scores.append(
                        ambiguous_score_lookup[(row["site"], row["image_id"])][1]
                    )
            evaluation_labels = np.asarray(
                [
                    int(row["label"] == "active_tb")
                    if row["label"] not in {label.value for label in AMBIGUOUS_LABELS}
                    else ambiguous_label
                    for row in validation_rows
                ],
                dtype=int,
            )
            evaluation_scores = np.asarray(validation_scores, dtype=float)
            threshold_metrics = evaluate_threshold(
                evaluation_scores, evaluation_labels, source_threshold
            )
            tier_result = TierAStudy().run(
                calibration_positive_scores=calibration_positive_scores,
                evaluation_scores=evaluation_scores,
                evaluation_labels=evaluation_labels,
                calibration_sizes=CALIBRATION_SIZES,
                repetitions=repetitions,
                seed=seed,
                split_unit=(
                    "official TBX11K train/validation image split; patient linkage "
                    "not verified; obsolete/unresolved labels remapped by scenario"
                ),
            )
            per_head["scenarios"][scenario] = {
                "ambiguous_label_mapping": ambiguous_label,
                "calibration_positive_count": len(calibration_positive_scores),
                "evaluation_positive_count": int(np.sum(evaluation_labels == 1)),
                "evaluation_negative_count": int(np.sum(evaluation_labels == 0)),
                "source_threshold_metrics": threshold_metrics.__dict__,
                "tier_a": tier_result.to_dict(),
            }
        scenario_results[head_name] = per_head

    result = {
        "study": "TBX11K obsolete/unresolved label sensitivity analysis",
        "encoder_key": encoder_key,
        "encoder_revision": ambiguous_metadata["revision"],
        "ambiguous_record_count": len(ambiguous_records),
        "ambiguous_counts_by_original_label": {
            label.value: sum(record.label is label for record in ambiguous_records)
            for label in sorted(AMBIGUOUS_LABELS, key=lambda value: value.value)
        },
        "ambiguous_counts_by_split": {
            split: sum(record.source_split == split for record in ambiguous_records)
            for split in ("train", "validation")
        },
        "exact_pixel_duplicate_aliases_excluded_from_evaluation": sum(
            row["site"] == "TBX11K"
            and row["source_split"] == "validation"
            and (row["site"], row["image_id"]) in duplicate_aliases
            for row in primary_rows
        ),
        "scenarios": scenario_results,
        "repetitions": repetitions,
        "seed": seed,
        "split_unit": "image-level; patient linkage not verified",
        "patient_level_claim": patient_level_supported(project_root),
        "pac_claim": False,
        "interpretation_limit": (
            "This is a sensitivity analysis under two extreme label mappings, not a "
            "claim that obsolete or unresolved findings are biologically TB-positive "
            "or TB-negative."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    result["output_path"] = str(output_path)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=("general", "chest_xray"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if arguments.repetitions < 1:
        parser.error("--repetitions must be positive")
    result = run_analysis(
        arguments.project_root,
        arguments.embedding_dir,
        arguments.encoder,
        arguments.output_dir,
        arguments.repetitions,
        arguments.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
