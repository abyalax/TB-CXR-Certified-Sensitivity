"""Empirical source-site threshold used only as the B0/K3 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EmpiricalThreshold:
    threshold: float
    target_sensitivity: float
    n_source_positives: int
    certified: bool = False


class EmpiricalSensitivityThreshold:
    """Choose a point-estimate threshold on source positives; gives no PAC claim."""

    def __init__(self, target_sensitivity: float = 0.90) -> None:
        if not 0 < target_sensitivity < 1:
            raise ValueError("target_sensitivity must be strictly between 0 and 1.")
        self.target_sensitivity = target_sensitivity

    def fit(self, source_positive_scores: Iterable[float]) -> EmpiricalThreshold:
        scores = np.asarray(tuple(source_positive_scores), dtype=float)
        if scores.ndim != 1 or scores.size == 0:
            raise ValueError("At least one source-positive score is required.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("source_positive_scores must be finite.")
        miss_count = int(np.floor((1 - self.target_sensitivity) * scores.size))
        threshold = float(np.sort(scores, kind="mergesort")[miss_count])
        return EmpiricalThreshold(
            threshold=threshold,
            target_sensitivity=self.target_sensitivity,
            n_source_positives=int(scores.size),
        )
