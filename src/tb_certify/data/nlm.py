"""Catalog the public Montgomery and Shenzhen CXR collections."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tb_certify.data.records import ClinicalClass, ImageRecord


@dataclass(frozen=True)
class NLMSiteLayout:
    """On-disk names used by one NLM chest X-ray collection."""

    directory: str
    filename_prefix: str


@dataclass(frozen=True)
class NLMSiteInventory:
    """Counts and linkage checks for images and their clinical readings."""

    site: str
    image_count: int
    image_suffix_counts: dict[str, int]
    clinical_reading_count: int
    clinical_reading_suffix_counts: dict[str, int]
    matched_image_reading_count: int
    images_without_reading_count: int
    readings_without_image_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NLMChestXrayCatalog:
    """Read image-level labels from documented NLM filename suffixes.

    The source README defines ``_0`` as normal and ``_1`` as abnormal/TB for
    these two collections. Clinical readings are checked by matching filename
    stems; their narrative text is not parsed as a substitute label. Patient
    identifiers are not inferred from image IDs.
    """

    SITE_LAYOUTS = {
        "Montgomery": NLMSiteLayout("montgomery", "MCUCXR"),
        "Shenzhen": NLMSiteLayout("shenzhen", "CHNCXR"),
    }

    def __init__(self, raw_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir)

    def _layout(self, site: str) -> NLMSiteLayout:
        for name, layout in self.SITE_LAYOUTS.items():
            if name.casefold() == site.casefold():
                return layout
        raise ValueError(f"Unknown NLM collection: {site}")

    def _files(self, site: str) -> tuple[list[Path], list[Path]]:
        layout = self._layout(site)
        site_dir = self.raw_dir / layout.directory
        image_dir = site_dir / "CXR_png"
        reading_dir = site_dir / "ClinicalReadings"
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        if not reading_dir.is_dir():
            raise FileNotFoundError(reading_dir)
        images = sorted(image_dir.glob("*.png"))
        readings = sorted(reading_dir.glob("*.txt"))
        if not images:
            raise ValueError(f"No NLM CXR images found in {image_dir}")
        return images, readings

    def inspect_site(self, site: str) -> NLMSiteInventory:
        """Summarize suffix labels and image-to-reading filename matches."""

        images, readings = self._files(site)
        image_ids = {path.stem for path in images}
        reading_ids = {path.stem for path in readings}
        image_suffixes = Counter(self._suffix(path.stem, site) for path in images)
        reading_suffixes = Counter(self._suffix(path.stem, site) for path in readings)
        return NLMSiteInventory(
            site=next(
                name
                for name in self.SITE_LAYOUTS
                if name.casefold() == site.casefold()
            ),
            image_count=len(images),
            image_suffix_counts=dict(sorted(image_suffixes.items())),
            clinical_reading_count=len(readings),
            clinical_reading_suffix_counts=dict(sorted(reading_suffixes.items())),
            matched_image_reading_count=len(image_ids & reading_ids),
            images_without_reading_count=len(image_ids - reading_ids),
            readings_without_image_count=len(reading_ids - image_ids),
        )

    def inspect(self) -> dict[str, NLMSiteInventory]:
        """Summarize both NLM collections."""

        return {
            site: self.inspect_site(site)
            for site in self.SITE_LAYOUTS
        }

    def records(self, site: str) -> tuple[ImageRecord, ...]:
        """Build labeled image records after verifying reading-file coverage."""

        images, readings = self._files(site)
        image_ids = {path.stem for path in images}
        reading_ids = {path.stem for path in readings}
        if image_ids != reading_ids:
            raise ValueError(
                f"{site} images and clinical readings do not match by filename "
                f"({len(image_ids - reading_ids)} missing readings, "
                f"{len(reading_ids - image_ids)} readings without images)."
            )

        canonical_site = next(
            name for name in self.SITE_LAYOUTS if name.casefold() == site.casefold()
        )
        records = []
        for path in images:
            suffix = self._suffix(path.stem, site)
            label = (
                ClinicalClass.HEALTHY if suffix == "0" else ClinicalClass.ACTIVE_TB
            )
            records.append(
                ImageRecord(
                    site=canonical_site,
                    image_id=path.stem,
                    label=label,
                    patient_id=None,
                    source_split="public_collection",
                    source_member=path.relative_to(self.raw_dir).as_posix(),
                )
            )
        return tuple(records)

    def _suffix(self, stem: str, site: str) -> str:
        prefix = self._layout(site).filename_prefix
        marker = f"{prefix}_"
        suffix = stem.rsplit("_", 1)[-1]
        if not stem.startswith(marker) or suffix not in {"0", "1"}:
            raise ValueError(f"Unexpected NLM image filename: {stem}")
        return suffix
