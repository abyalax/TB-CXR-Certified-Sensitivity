"""Small immutable domain objects used across data and experiment modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ClinicalClass(str, Enum):
    """Image-level categories before applying the active-TB study policy."""

    ACTIVE_TB = "active_tb"
    HEALTHY = "healthy"
    NON_TB_DISEASE = "non_tb_disease"
    OBSOLETE_TB = "obsolete_tb"
    UNRESOLVED_TB = "unresolved_tb"

    @property
    def binary_label(self) -> int | None:
        """Return 1/0 for the primary study or None for excluded TB findings."""

        if self is ClinicalClass.ACTIVE_TB:
            return 1
        if self in (ClinicalClass.HEALTHY, ClinicalClass.NON_TB_DISEASE):
            return 0
        return None

    @property
    def is_tb_positive(self) -> bool:
        return self.binary_label == 1


@dataclass(frozen=True)
class LesionBox:
    """A lesion box in the source image's pixel coordinate system."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y, self.width, self.height)):
            raise ValueError("Lesion box coordinates must be finite.")
        if self.x < 0 or self.y < 0:
            raise ValueError("Lesion box coordinates must be non-negative.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Lesion box width and height must be positive.")


@dataclass(frozen=True)
class ROIAnnotation:
    """One lesion annotation; image_id is not assumed to be a patient ID."""

    site: str
    image_id: str
    box: LesionBox
    label: str


@dataclass(frozen=True)
class ImageRecord:
    """Image-level record with an optional, explicitly sourced patient key."""

    site: str
    image_id: str
    label: ClinicalClass | None
    patient_id: str | None = None
    source_split: str | None = None
    source_member: str | None = None

    def __post_init__(self) -> None:
        if not self.site.strip():
            raise ValueError("site must not be empty.")
        if not self.image_id.strip():
            raise ValueError("image_id must not be empty.")
        if self.patient_id is not None and not self.patient_id.strip():
            raise ValueError("patient_id must be non-empty when provided.")

