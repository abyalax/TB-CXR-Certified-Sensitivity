"""Threshold metrics with explicit positive-class and score semantics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    true_positives: int
    false_negatives: int
    true_negatives: int
    false_positives: int
    sensitivity: float | None
    specificity: float | None


def evaluate_threshold(
    scores: Iterable[float], labels: Iterable[int], threshold: float
) -> ThresholdMetrics:
    """Evaluate a greater-than-or-equal TB-positive rule on fully labeled data."""

    score_array = np.asarray(tuple(scores), dtype=float)
    label_array = np.asarray(tuple(labels), dtype=int)
    if score_array.ndim != 1 or label_array.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional.")
    if score_array.size == 0 or score_array.size != label_array.size:
        raise ValueError("scores and labels must be non-empty and have equal length.")
    if not np.all(np.isfinite(score_array)) or not np.isfinite(threshold):
        raise ValueError("scores and threshold must be finite.")
    if not np.isin(label_array, [0, 1]).all():
        raise ValueError("labels must use 1 for active TB and 0 for non-TB.")

    predicted_positive = score_array >= threshold
    actual_positive = label_array == 1
    true_positives = int(np.sum(predicted_positive & actual_positive))
    false_negatives = int(np.sum(~predicted_positive & actual_positive))
    true_negatives = int(np.sum(~predicted_positive & ~actual_positive))
    false_positives = int(np.sum(predicted_positive & ~actual_positive))
    n_positive = true_positives + false_negatives
    n_negative = true_negatives + false_positives
    return ThresholdMetrics(
        threshold=float(threshold),
        true_positives=true_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        sensitivity=true_positives / n_positive if n_positive else None,
        specificity=true_negatives / n_negative if n_negative else None,
    )
