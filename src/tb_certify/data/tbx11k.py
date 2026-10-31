"""Non-extracting catalog for the public TBX11K archive splits."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

from tb_certify.data.records import ClinicalClass, ImageRecord


class TBX11KSchemaError(ValueError):
    """Raised when archive manifests contradict their expected structure."""


@dataclass(frozen=True)
class TBX11KInventory:
    archive_path: str
    image_counts_by_directory: dict[str, int]
    public_split_counts: dict[str, dict[str, int]]
    clinical_label_counts_by_split: dict[str, dict[str, int]]
    annotation_box_counts: dict[str, int]
    public_trainval_count: int
    test_directory_count: int
    test_labels_public: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TBX11KArchiveCatalog:
    """Inspect official TBX11K train/validation manifests without extracting.

    The archive README states that the competition test ground truth is not
    released. The ``all_*`` lists also combine TBX11K with external datasets;
    this catalog deliberately uses only ``TBX11K_*`` lists to prevent site
    contamination. Public folder labels are kept as three classes so the
    healthy-only and healthy-plus-non-TB negative-control spectra remain
    distinguishable.
    """

    ROOT = "TBX11K"
    SPLIT_MEMBERS = {
        "train": "TBX11K/lists/TBX11K_train.txt",
        "validation": "TBX11K/lists/TBX11K_val.txt",
        "trainval": "TBX11K/lists/TBX11K_trainval.txt",
    }
    ANNOTATION_MEMBER = "TBX11K/annotations/json/TBX11K_trainval.json"
    CATEGORY_LABELS = {
        "health": ClinicalClass.HEALTHY,
        "sick": ClinicalClass.NON_TB_DISEASE,
    }
    TB_ANNOTATION_CATEGORIES = {
        "ActiveTuberculosis": ClinicalClass.ACTIVE_TB,
        "ObsoletePulmonaryTuberculosis": ClinicalClass.OBSOLETE_TB,
        "PulmonaryTuberculosis": ClinicalClass.UNRESOLVED_TB,
    }
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    def __init__(self, archive_path: str | Path) -> None:
        self.archive_path = Path(archive_path)
        if not self.archive_path.is_file():
            raise FileNotFoundError(self.archive_path)

    def _open_archive(self) -> ZipFile:
        try:
            return ZipFile(self.archive_path, mode="r")
        except BadZipFile as error:
            raise TBX11KSchemaError(
                f"Not a readable ZIP archive: {self.archive_path}"
            ) from error

    @staticmethod
    def _split_rows(archive: ZipFile, split: str, members: set[str]) -> tuple[str, ...]:
        try:
            member = TBX11KArchiveCatalog.SPLIT_MEMBERS[split]
        except KeyError as error:
            raise ValueError(f"Unknown TBX11K split: {split}") from error
        if member not in members:
            raise TBX11KSchemaError(f"Missing official split list: {member}")

        with archive.open(member, "r") as stream:
            text_stream = io.TextIOWrapper(stream, encoding="utf-8-sig")
            rows = tuple(line.strip().replace("\\", "/") for line in text_stream)
        rows = tuple(row for row in rows if row)
        if len(rows) != len(set(rows)):
            raise TBX11KSchemaError(f"Duplicate image path in {member}")

        for row in rows:
            relative = PurePosixPath(row)
            if relative.is_absolute() or ".." in relative.parts:
                raise TBX11KSchemaError(f"Unsafe image path in {member}: {row}")
            if len(relative.parts) < 2:
                raise TBX11KSchemaError(f"Invalid image path in {member}: {row}")
            if relative.parts[0] not in (
                *TBX11KArchiveCatalog.CATEGORY_LABELS,
                "tb",
            ):
                raise TBX11KSchemaError(
                    f"Unexpected public class '{relative.parts[0]}' in {member}"
                )
            archive_image = f"TBX11K/imgs/{relative.as_posix()}"
            if archive_image not in members:
                raise TBX11KSchemaError(
                    f"Split references an image missing from the archive: {archive_image}"  # noqa: E501
                )
        return rows

    @classmethod
    def _annotation_index(
        cls, archive: ZipFile, members: set[str]
    ) -> tuple[dict[str, ClinicalClass], dict[str, int]]:
        if cls.ANNOTATION_MEMBER not in members:
            raise TBX11KSchemaError(
                f"Missing public image annotations: {cls.ANNOTATION_MEMBER}"
            )
        with archive.open(cls.ANNOTATION_MEMBER, "r") as stream:
            document = json.load(stream)

        categories = {
            int(category["id"]): str(category["name"])
            for category in document.get("categories", [])
        }
        missing_categories = set(cls.TB_ANNOTATION_CATEGORIES).difference(
            categories.values()
        )
        if missing_categories:
            raise TBX11KSchemaError(
                f"TBX11K lesion taxonomy is incomplete: {sorted(missing_categories)}"
            )
        category_ids = {
            category_id: cls.TB_ANNOTATION_CATEGORIES[name]
            for category_id, name in categories.items()
            if name in cls.TB_ANNOTATION_CATEGORIES
        }

        annotation_rows = document.get("annotations", [])
        unknown_category_ids = {
            int(annotation["category_id"])
            for annotation in annotation_rows
            if int(annotation["category_id"]) not in category_ids
        }
        if unknown_category_ids:
            raise TBX11KSchemaError(
                "Annotations use an unmapped category ID: "
                f"{sorted(unknown_category_ids)}"
            )

        image_names = {
            int(image["id"]): str(image["file_name"]).replace("\\", "/")
            for image in document.get("images", [])
        }
        unknown_image_ids = {
            int(annotation["image_id"])
            for annotation in annotation_rows
            if int(annotation["image_id"]) not in image_names
        }
        if unknown_image_ids:
            raise TBX11KSchemaError(
                "Annotations reference an unknown image ID: "
                f"{sorted(unknown_image_ids)[:10]}"
            )
        annotations_by_image: dict[int, set[ClinicalClass]] = {}
        box_counts: Counter[str] = Counter()
        for annotation in annotation_rows:
            category_id = int(annotation["category_id"])
            category = category_ids.get(category_id)
            if category is None:
                continue
            image_id = int(annotation["image_id"])
            annotations_by_image.setdefault(image_id, set()).add(category)
            box_counts[category.value] += 1

        image_labels: dict[str, ClinicalClass] = {}
        for image_id, image_name in image_names.items():
            path = PurePosixPath(image_name)
            category = path.parts[0] if path.parts else ""
            if category in cls.CATEGORY_LABELS:
                image_labels[image_name] = cls.CATEGORY_LABELS[category]
                continue
            if category != "tb":
                continue
            lesion_classes = annotations_by_image.get(image_id, set())
            if ClinicalClass.ACTIVE_TB in lesion_classes:
                image_labels[image_name] = ClinicalClass.ACTIVE_TB
            elif ClinicalClass.OBSOLETE_TB in lesion_classes:
                image_labels[image_name] = ClinicalClass.OBSOLETE_TB
            else:
                image_labels[image_name] = ClinicalClass.UNRESOLVED_TB

        return image_labels, dict(sorted(box_counts.items()))

    @staticmethod
    def _clinical_counts(
        rows: tuple[str, ...], image_labels: dict[str, ClinicalClass]
    ) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for row in rows:
            counts[image_labels[row].value] += 1
        return dict(sorted(counts.items()))

    def inspect(self) -> TBX11KInventory:
        with self._open_archive() as archive:
            names = set(archive.namelist())
            image_labels, annotation_box_counts = self._annotation_index(archive, names)
            directory_counts: Counter[str] = Counter()
            for name in names:
                path = PurePosixPath(name)
                if (
                    len(path.parts) >= 4
                    and path.parts[:2] == (self.ROOT, "imgs")
                    and path.suffix.lower() in self.IMAGE_SUFFIXES
                ):
                    directory_counts[path.parts[2]] += 1

            split_counts: dict[str, dict[str, int]] = {}
            clinical_counts: dict[str, dict[str, int]] = {}
            split_rows = {
                split: self._split_rows(archive, split, names)
                for split in self.SPLIT_MEMBERS
            }
            for split, rows in split_rows.items():
                categories = Counter(PurePosixPath(row).parts[0] for row in rows)
                split_counts[split] = dict(sorted(categories.items()))
                clinical_counts[split] = self._clinical_counts(rows, image_labels)

            trainval = set(split_rows["trainval"])
            train_and_validation = set(split_rows["train"]) | set(
                split_rows["validation"]
            )
            warnings: list[str] = []
            if trainval != train_and_validation:
                warnings.append(
                    "TBX11K_trainval does not equal the union of official train and "
                    "validation lists; inspect the split manifests before training."
                )

            allowed_public_classes = set(self.CATEGORY_LABELS) | {"tb"}
            unknown_public_classes = (
                set(split_counts["trainval"]) - allowed_public_classes
            )
            if unknown_public_classes:
                raise TBX11KSchemaError(
                    f"Unknown public classes: {sorted(unknown_public_classes)}"
                )

            test_count = directory_counts.get("test", 0)
            if test_count:
                warnings.append(
                    "The test directory is excluded: the bundled README says its "
                    "ground-truth labels are not released."
                )
            if any(name.startswith("TBX11K/lists/all_") for name in names):
                warnings.append(
                    "The all_* lists combine TBX11K and external-site images; this "
                    "catalog does not use them."
                )

            return TBX11KInventory(
                archive_path=str(self.archive_path.resolve()),
                image_counts_by_directory=dict(sorted(directory_counts.items())),
                public_split_counts=split_counts,
                clinical_label_counts_by_split=clinical_counts,
                annotation_box_counts=annotation_box_counts,
                public_trainval_count=len(split_rows["trainval"]),
                test_directory_count=test_count,
                test_labels_public=False,
                warnings=tuple(warnings),
            )

    def public_records(self, split: str = "trainval") -> tuple[ImageRecord, ...]:
        """Return only images named by the official TBX11K split list."""

        with self._open_archive() as archive:
            members = set(archive.namelist())
            image_labels, _ = self._annotation_index(archive, members)
            rows = self._split_rows(archive, split, members)
            split_by_image: dict[str, str] = {}
            if split == "trainval":
                for component in ("train", "validation"):
                    for image in self._split_rows(archive, component, members):
                        split_by_image[image] = component
            records: list[ImageRecord] = []
            for relative_path in rows:
                relative = PurePosixPath(relative_path)
                records.append(
                    ImageRecord(
                        site="TBX11K",
                        image_id=relative.name,
                        label=image_labels[relative.as_posix()],
                        patient_id=None,
                        source_split=split_by_image.get(relative_path, split),
                        source_member=f"TBX11K/imgs/{relative.as_posix()}",
                    )
                )
            return tuple(records)

    def read_image_bytes(self, record: ImageRecord) -> bytes:
        """Read one vetted public image member without expanding the full ZIP."""

        with self.open_image_reader() as reader:
            return reader.read_image_bytes(record)

    def open_image_reader(self) -> TBX11KImageReader:
        """Open a reusable reader for processing many images without extraction."""

        return TBX11KImageReader(self.archive_path)

    def primary_analysis_records(
        self, split: str = "trainval"
    ) -> tuple[ImageRecord, ...]:  # noqa: E501
        """Filter out obsolete or unresolved TB findings from the binary study."""

        return tuple(
            record
            for record in self.public_records(split=split)
            if record.label is not None and record.label.binary_label is not None
        )

    def write_manifest(self, destination: str | Path, split: str = "trainval") -> Path:
        """Write image-level labels; no verified patient crosswalk is available."""

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        records = self.public_records(split=split)
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "site",
                    "image_id",
                    "patient_id",
                    "label",
                    "source_split",
                    "source_member",
                ),
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "site": record.site,
                        "image_id": record.image_id,
                        "patient_id": record.patient_id or "",
                        "label": record.label.value if record.label else "",
                        "source_split": record.source_split or "",
                        "source_member": record.source_member or "",
                    }
                )
        return target


class TBX11KImageReader:
    """Context-managed image access that keeps the archive open for one run."""

    ROOT = "TBX11K"

    def __init__(self, archive_path: str | Path) -> None:
        self.archive_path = Path(archive_path)
        self._archive: ZipFile | None = None

    def __enter__(self) -> TBX11KImageReader:
        if self._archive is not None:
            raise RuntimeError("TBX11KImageReader is already open.")
        self._archive = ZipFile(self.archive_path, mode="r")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def read_image_bytes(self, record: ImageRecord) -> bytes:
        if self._archive is None:
            raise RuntimeError("Use TBX11KImageReader inside a with block.")
        member = record.source_member
        if record.site != "TBX11K" or not member:
            raise ValueError("A TBX11K record with a source_member is required.")
        normalized = PurePosixPath(member)
        if (
            normalized.is_absolute()
            or ".." in normalized.parts
            or normalized.parts[:2] != (self.ROOT, "imgs")
        ):
            raise ValueError("The image record contains an unsafe archive path.")
        try:
            with self._archive.open(normalized.as_posix(), "r") as stream:
                return stream.read()
        except KeyError as error:
            raise TBX11KSchemaError(
                f"Image member is missing: {normalized.as_posix()}"
            ) from error
