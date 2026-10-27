"""Describe source-threshold sensitivity by annotated lesion-size stratum.

The Montgomery and Shenzhen ROI files annotate only positive lesions. This
script therefore reports a descriptive analysis on their matched, annotated
positive-image subsets; it does not infer severity for unannotated images or
patient-level effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scripts.run_primary_real_study import _fit_head, _read_csv, _scores, _source_split

from tb_certify.data.roi import ROIAnnotationCsvReader
from tb_certify.evaluation.binomial import clopper_pearson_interval


def _max_relative_lesion_area(
    project_root: Path, site: str, image_id: str, boxes: list[tuple[float, float]]
) -> float:
    image_path = (
        project_root / "data" / "raw" / site.lower() / "CXR_png" / f"{image_id}.png"
    )
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        width, height = image.size
    return max(area for area, _ in boxes) / (width * height)


def _strata(values: dict[str, float]) -> dict[str, set[str]]:
    ordered = sorted(values, key=values.__getitem__)
    chunks = np.array_split(np.asarray(ordered, dtype=object), 3)
    return {
        "small": set(chunks[0].tolist()),
        "medium": set(chunks[1].tolist()),
        "large": set(chunks[2].tolist()),
    }


def run_analysis(
    project_root: Path, embedding_dir: Path, output_path: Path
) -> dict[str, Any]:
    rows = _read_csv(embedding_dir / "manifest.csv")
    features = np.load(embedding_dir / "embeddings.npy", mmap_mode="r")
    metadata = json.loads((embedding_dir / "metadata.json").read_text(encoding="utf-8"))
    if features.shape[0] != len(rows):
        raise ValueError("Embedding rows must align with the manifest.")

    target_site = "Shenzhen"
    source_sites = ("Montgomery", "TBX11K")
    train_indices, validation_indices, _ = _source_split(rows, source_sites, seed=42)
    target_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["site"] == target_site],
        dtype=int,
    )
    reader = ROIAnnotationCsvReader()
    annotations = reader.read(
        project_root / "data" / "raw" / "shenzhen" / "shenzhen_consensus_roi.csv",
        target_site,
    )
    box_map: dict[str, list[tuple[float, float]]] = {}
    for annotation in annotations:
        image_id = Path(annotation.image_id).stem
        box_map.setdefault(image_id, []).append(
            (annotation.box.width * annotation.box.height, annotation.box.height)
        )
    target_by_id = {rows[index]["image_id"]: int(index) for index in target_indices}
    missing = set(box_map).difference(target_by_id)
    if missing:
        raise ValueError(
            f"ROI image IDs do not match the target manifest: {sorted(missing)}"
        )
    if any(
        rows[target_by_id[image_id]]["label"] != "active_tb" for image_id in box_map
    ):
        raise ValueError(
            "ROI annotations must only be matched to active-TB target images."
        )

    relative_areas = {
        image_id: _max_relative_lesion_area(project_root, "shenzhen", image_id, boxes)
        for image_id, boxes in box_map.items()
    }
    strata = _strata(relative_areas)
    results: dict[str, Any] = {
        "study": "annotated-lesion-size sensitivity analysis",
        "target_site": target_site,
        "source_sites": list(source_sites),
        "encoder": metadata,
        "seed": 42,
        "analysis_unit": "annotated positive patient (one image per patient)",
        "patient_level_claim": True,
        "interpretation_limit": (
            "ROI tables contain only lesion-positive annotations. Results describe the "
            "68 matched Shenzhen patients (one image per patient), not all target "
            "positives or a clinical severity effect."
        ),
        "lesion_area_definition": (
            "maximum annotated box area divided by original image area"
        ),
        "annotated_positive_images": len(relative_areas),
        "strata": {
            name: {
                "count": len(image_ids),
                "minimum_relative_area": float(
                    min(relative_areas[item] for item in image_ids)
                ),
                "maximum_relative_area": float(
                    max(relative_areas[item] for item in image_ids)
                ),
            }
            for name, image_ids in strata.items()
        },
        "heads": {},
    }
    for head_name in ("logistic", "mlp"):
        head, _, _, threshold = _fit_head(
            head_name,
            features,
            rows,
            train_indices,
            validation_indices,
            target_site,
            source_sites,
            42,
        )
        all_scores = _scores(head, head_name, features[target_indices])
        score_by_id = {
            rows[index]["image_id"]: float(score)
            for index, score in zip(target_indices, all_scores, strict=True)
        }
        head_summary: dict[str, Any] = {
            "source_threshold": float(threshold),
            "by_lesion_size": {},
        }
        for name, image_ids in strata.items():
            detected = sum(score_by_id[image_id] >= threshold for image_id in image_ids)
            interval = clopper_pearson_interval(
                detected, len(image_ids), confidence=0.95
            )
            head_summary["by_lesion_size"][name] = {
                "detected": int(detected),
                "total": len(image_ids),
                "sensitivity": detected / len(image_ids),
                "sensitivity_interval": interval.__dict__,
            }
        results["heads"][head_name] = head_summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_analysis(
        arguments.project_root.resolve(),
        arguments.embedding_dir.resolve(),
        arguments.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
