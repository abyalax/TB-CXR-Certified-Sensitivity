"""Compare RAD-DINO's public training file list with a local image manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

MODEL_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
TRAINING_LIST_URL = (
    "https://huggingface.co/microsoft/rad-dino/resolve/"
    f"{MODEL_REVISION}/training_images.csv"
)


def _stem_and_name(value: str) -> tuple[str, str]:
    path = PurePosixPath(value.replace("\\", "/"))
    return path.stem.casefold(), path.name.casefold()


def audit_training_overlap(training_list: Path, manifest_path: Path) -> dict[str, Any]:
    """Summarize metadata-level overlap; image pixels are not compared."""

    with manifest_path.open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    local_ids = {row["image_id"].casefold() for row in manifest}
    local_stems: set[str] = set()
    local_basenames: set[str] = set()
    for row in manifest:
        stem, basename = _stem_and_name(row["source_member"])
        local_stems.add(stem)
        local_basenames.add(basename)

    training_ids: set[str] = set()
    training_paths: set[str] = set()
    training_stems: set[str] = set()
    training_basenames: set[str] = set()
    source_counts: Counter[str] = Counter()
    row_widths: Counter[int] = Counter()
    record_count = 0
    with training_list.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != ["image_id", "rel_image_path"]:
            raise ValueError(f"Unexpected training list columns: {header!r}")
        for row in reader:
            record_count += 1
            row_widths[len(row)] += 1
            if len(row) != 2:
                raise ValueError(
                    f"Malformed training list row {record_count + 1}: "
                    f"expected 2 fields, found {len(row)}"
                )
            image_id, relative_path = row
            image_id = image_id.casefold()
            stem, basename = _stem_and_name(relative_path)
            training_paths.add(relative_path.casefold())
            parts = PurePosixPath(relative_path).parts
            source_counts[parts[0] if parts else "<empty>"] += 1
            training_ids.add(image_id)
            training_stems.add(stem)
            training_basenames.add(basename)

    with training_list.open("rb") as stream:
        list_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

    return {
        "model": "microsoft/rad-dino",
        "model_revision": MODEL_REVISION,
        "training_list_url": TRAINING_LIST_URL,
        "training_list_size_bytes": training_list.stat().st_size,
        "training_list_sha256": list_sha256,
        "training_list_header": ["image_id", "rel_image_path"],
        "training_image_count": record_count,
        "unique_training_image_ids": len(training_ids),
        "unique_training_paths": len(training_paths),
        "training_rows_by_source": dict(sorted(source_counts.items())),
        "training_row_widths": dict(sorted(row_widths.items())),
        "local_manifest_rows": len(manifest),
        "metadata_overlap": {
            "image_id_matches": len(training_ids & local_ids),
            "image_id_vs_source_stem_matches": len(training_ids & local_stems),
            "path_stem_matches": len(training_stems & (local_stems | local_ids)),
            "basename_matches": len(training_basenames & local_basenames),
            "pixel_level_duplicate_check": "not_performed",
        },
        "limitations": [
            "The public list contains image IDs and paths, not source image hashes.",
            "Renamed or transformed copies cannot be excluded by this comparison.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_list", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = audit_training_overlap(arguments.training_list, arguments.manifest)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
