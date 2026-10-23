"""Reader for the Montgomery and Shenzhen consensus lesion ROI tables."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tb_certify.data.records import LesionBox, ROIAnnotation


class AnnotationSchemaError(ValueError):
    """Raised when an ROI CSV does not match the expected source schema."""


@dataclass(frozen=True)
class ROISummary:
    site: str
    annotation_count: int
    unique_image_count: int
    label_counts: dict[str, int]


class ROIAnnotationCsvReader:
    """Read lesion boxes without inferring image-level negative labels.

    The source column is named ``patientId`` but contains image filenames in
    the supplied files. It is intentionally stored as ``image_id``; these
    tables annotate positive lesions and do not enumerate the negative cohort.
    """

    REQUIRED_COLUMNS = (
        "patientId",
        "x_dis",
        "y_dis",
        "width_dis",
        "height_dis",
        "Labels",
    )

    def read(self, path: str | Path, site: str) -> tuple[ROIAnnotation, ...]:
        csv_path = Path(path)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)

        annotations: list[ROIAnnotation] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = tuple(reader.fieldnames or ())
            missing = set(self.REQUIRED_COLUMNS).difference(columns)
            if missing:
                raise AnnotationSchemaError(
                    f"{csv_path} is missing required columns: {sorted(missing)}"
                )

            for row_number, row in enumerate(reader, start=2):
                image_id = (row.get("patientId") or "").strip()
                label = (row.get("Labels") or "").strip()
                if not image_id or not label:
                    raise AnnotationSchemaError(
                        f"{csv_path}:{row_number} has an empty image ID or label."
                    )
                try:
                    box = LesionBox(
                        x=float(row["x_dis"]),
                        y=float(row["y_dis"]),
                        width=float(row["width_dis"]),
                        height=float(row["height_dis"]),
                    )
                except (TypeError, ValueError) as error:
                    raise AnnotationSchemaError(
                        f"{csv_path}:{row_number} contains invalid box coordinates."
                    ) from error
                annotations.append(
                    ROIAnnotation(site=site, image_id=image_id, box=box, label=label)
                )

        return tuple(annotations)

    @staticmethod
    def summarize(site: str, annotations: Iterable[ROIAnnotation]) -> ROISummary:
        rows = tuple(annotations)
        return ROISummary(
            site=site,
            annotation_count=len(rows),
            unique_image_count=len({row.image_id for row in rows}),
            label_counts=dict(sorted(Counter(row.label for row in rows).items())),
        )
