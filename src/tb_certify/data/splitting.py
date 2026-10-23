"""Patient-grouped splits that fail closed when IDs are unavailable."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from tb_certify.data.records import ImageRecord


class PatientIdentifierRequiredError(ValueError):
    """Raised rather than silently treating image IDs as patient IDs."""


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[ImageRecord, ...]
    test: tuple[ImageRecord, ...]


class PatientLevelSplitter:
    """Create a stratified split over explicit patient IDs, not image rows."""

    def __init__(self, test_fraction: float = 0.5, seed: int = 42) -> None:
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1.")
        self.test_fraction = test_fraction
        self.seed = seed

    def split(self, records: Sequence[ImageRecord]) -> DatasetSplit:
        if len(records) < 2:
            raise ValueError("At least two image records are required to split.")
        if any(record.patient_id is None for record in records):
            raise PatientIdentifierRequiredError(
                "Patient-level splitting requires verified patient IDs. "
                "Image IDs are not substituted for patient IDs."
            )
        if any(record.label is None for record in records):
            raise ValueError("All records need image-level labels before splitting.")
        if any(record.label.binary_label is None for record in records):
            raise ValueError(
                "Records without an active-TB or non-TB binary label must be "
                "excluded before splitting."
            )

        group_labels: dict[tuple[str, str], set[str]] = {}
        records_by_patient: dict[tuple[str, str], list[ImageRecord]] = {}
        for record in records:
            assert record.patient_id is not None
            assert record.label is not None
            patient_key = (record.site, record.patient_id)
            group_labels.setdefault(patient_key, set()).add(
                str(record.label.binary_label)
            )
            records_by_patient.setdefault(patient_key, []).append(record)

        conflicting = [
            patient_key
            for patient_key, labels in group_labels.items()
            if len(labels) != 1
        ]
        if conflicting:
            raise ValueError(
                "A patient has conflicting image-level labels; resolve the "
                f"label policy before splitting ({len(conflicting)} patients)."
            )
        patient_keys = sorted(records_by_patient)
        labels = [next(iter(group_labels[key])) for key in patient_keys]
        train_ids, test_ids = train_test_split(
            patient_keys,
            test_size=self.test_fraction,
            random_state=self.seed,
            stratify=labels,
        )
        train_set, test_set = set(train_ids), set(test_ids)
        train_records = tuple(
            record for record in records if (record.site, record.patient_id) in train_set
        )
        test_records = tuple(
            record for record in records if (record.site, record.patient_id) in test_set
        )
        if train_set.intersection(test_set):
            raise RuntimeError("Internal error: patient groups overlap after split.")
        return DatasetSplit(train=train_records, test=test_records)
