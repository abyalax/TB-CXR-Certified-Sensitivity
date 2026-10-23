"""Build reproducible image caches for the primary binary study cohort."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from tb_certify.data.nlm import NLMChestXrayCatalog
from tb_certify.data.preprocessing import CXRImagePreprocessor, CXRPreprocessingConfig
from tb_certify.data.records import ImageRecord
from tb_certify.data.tbx11k import TBX11KArchiveCatalog


@dataclass(frozen=True)
class PreprocessingSummary:
    """Counts and output paths for a completed cache build."""

    output_dir: str
    manifest_path: str
    image_count: int
    label_counts_by_site: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PrimaryCohortPreprocessor:
    """Create a lossless PNG cache and provenance manifest for primary data."""

    MANIFEST_FIELDS = (
        "site",
        "image_id",
        "patient_id",
        "label",
        "source_split",
        "source_member",
        "cache_path",
        "height",
        "width",
        "channels",
        "cache_sha256",
    )

    def __init__(
        self,
        raw_dir: str | Path,
        output_dir: str | Path,
        config: CXRPreprocessingConfig | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir == self.raw_dir or self.raw_dir in self.output_dir.parents:
            raise ValueError("The processed cache must be outside data/raw/.")
        self.config = config or CXRPreprocessingConfig()
        self.preprocessor = CXRImagePreprocessor(self.config)

    def _records(self) -> tuple[ImageRecord, ...]:
        nlm_catalog = NLMChestXrayCatalog(self.raw_dir)
        records = [
            *nlm_catalog.records("Montgomery"),
            *nlm_catalog.records("Shenzhen"),
        ]
        tbx_catalog = TBX11KArchiveCatalog(self.raw_dir / "TBX11K.zip")
        records.extend(tbx_catalog.primary_analysis_records(split="trainval"))
        return tuple(records)

    def _cache_relative_path(self, record: ImageRecord) -> Path:
        if not record.source_member:
            raise ValueError(f"Missing source path for {record.site}/{record.image_id}")
        identity = f"{record.site}/{record.source_member}".encode()
        cache_key = hashlib.sha256(identity).hexdigest()[:24]
        return Path("images") / record.site.casefold() / f"{cache_key}.png"

    def run(self) -> PreprocessingSummary:
        """Preprocess every primary cohort image and publish its manifest."""

        records = self._records()
        if not records:
            raise ValueError("The primary image cohort is empty.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "manifest.csv"
        temporary_manifest = self.output_dir / "manifest.csv.tmp"
        label_counts: dict[str, Counter[str]] = defaultdict(Counter)
        cache_paths: set[Path] = set()

        try:
            with temporary_manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.MANIFEST_FIELDS)
                writer.writeheader()
                with TBX11KArchiveCatalog(
                    self.raw_dir / "TBX11K.zip"
                ).open_image_reader() as archive_reader:
                    for record in records:
                        if record.label is None or record.label.binary_label is None:
                            raise ValueError(
                                "Primary cohort contains an excluded or unlabeled "
                                "record: "
                                f"{record.site}/{record.image_id}"
                            )
                        cache_relative = self._cache_relative_path(record)
                        if cache_relative in cache_paths:
                            raise ValueError(f"Duplicate cache key: {cache_relative}")
                        cache_paths.add(cache_relative)
                        cache_path = self.output_dir / cache_relative
                        cache_path.parent.mkdir(parents=True, exist_ok=True)

                        if record.site == "TBX11K":
                            source_bytes = archive_reader.read_image_bytes(record)
                            image_array = self.preprocessor.process_bytes(source_bytes)
                        else:
                            source_path = self.raw_dir / (record.source_member or "")
                            image_array = self.preprocessor.process_path(source_path)

                        payload_stream = BytesIO()
                        Image.fromarray(image_array).save(
                            payload_stream, format="PNG", optimize=False
                        )
                        payload = payload_stream.getvalue()
                        temporary_cache = cache_path.with_suffix(".png.tmp")
                        temporary_cache.write_bytes(payload)
                        os.replace(temporary_cache, cache_path)

                        cache_path_in_manifest = cache_relative.as_posix()
                        writer.writerow(
                            {
                                "site": record.site,
                                "image_id": record.image_id,
                                "patient_id": record.patient_id or "",
                                "label": record.label.value,
                                "source_split": record.source_split or "",
                                "source_member": record.source_member,
                                "cache_path": cache_path_in_manifest,
                                "height": image_array.shape[0],
                                "width": image_array.shape[1],
                                "channels": image_array.shape[2],
                                "cache_sha256": hashlib.sha256(payload).hexdigest(),
                            }
                        )
                        label_counts[record.site][record.label.value] += 1

            os.replace(temporary_manifest, manifest_path)
        except Exception:
            temporary_manifest.unlink(missing_ok=True)
            raise

        metadata = {
            "preprocessing": asdict(self.config),
            "pixel_format": "uint8 RGB PNG",
            "crop_policy": "full_image_no_crop",
            "label_policy": {
                "Montgomery/Shenzhen": "documented filename suffix",
                "TBX11K": "ActiveTuberculosis box; obsolete/unresolved excluded",
            },
            "patient_id_policy": "blank unless a verified patient crosswalk exists",
            "record_count": len(records),
            "successful_images": len(records),
            "decode_failures": 0,
            "random_seed": "not_applicable_deterministic_preprocessing",
            "label_counts_by_site": {
                site: dict(sorted(counts.items()))
                for site, counts in sorted(label_counts.items())
            },
        }
        metadata_path = self.output_dir / "preprocessing.json"
        temporary_metadata = self.output_dir / "preprocessing.json.tmp"
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_metadata, metadata_path)
        return PreprocessingSummary(
            output_dir=str(self.output_dir),
            manifest_path=str(manifest_path),
            image_count=len(records),
            label_counts_by_site=metadata["label_counts_by_site"],
        )
