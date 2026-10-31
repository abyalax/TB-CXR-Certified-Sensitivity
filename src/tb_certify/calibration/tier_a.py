"""Exact target-only PAC sensitivity certification (Tier A)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import binom


class CertificateStatus(str, Enum):
    CERTIFIED = "certified"
    REFUSE = "refuse"


@dataclass(frozen=True)
class SensitivityCertificate:
    status: CertificateStatus
    threshold: float | None
    n_positive: int
    allowed_misses: int | None
    target_fnr: float
    failure_probability: float
    reason: str | None = None

    @property
    def issued(self) -> bool:
        return self.status is CertificateStatus.CERTIFIED


class TierACalibrator:
    """Certify population sensitivity using target-site positive scores only.

    Scores must increase with evidence for active TB, with one score per
    independent sampling unit. For ``n`` i.i.d., representative target-site
    TB-positive calibration units, the method
    selects the ``(k* + 1)``-st score in ascending order, where ``k*`` is the
    largest miss count whose Binomial(n, alpha) lower tail is at most delta.
    No threshold is issued below the exact finite-sample certification floor.
    Validity also requires a fixed model/score rule, accurate reference labels,
    and a stationary target population between calibration and use. Tied
    scores at the selected order statistic are all included by the ``>=``
    prediction rule, which can only reduce the realized calibration misses.
    """

    def __init__(self, target_fnr: float = 0.10, failure_probability: float = 0.05):
        self._validate_probability(target_fnr, "target_fnr")
        self._validate_probability(failure_probability, "failure_probability")
        self.target_fnr = target_fnr
        self.failure_probability = failure_probability

    @staticmethod
    def _validate_probability(value: float, name: str) -> None:
        if not 0 < value < 1:
            raise ValueError(f"{name} must be strictly between 0 and 1.")

    def allowed_misses(self, n_positive: int) -> int | None:
        """Return k*; ``None`` means no distribution-free certificate exists."""

        if n_positive < 1:
            return None
        feasible = [
            k
            for k in range(n_positive)
            if binom.cdf(k, n_positive, self.target_fnr) <= self.failure_probability
        ]
        return max(feasible) if feasible else None

    @property
    def minimum_positive_count(self) -> int:
        """Smallest n with any certifiable order statistic for these settings."""

        n_positive = 1
        while self.allowed_misses(n_positive) is None:
            n_positive += 1
        return n_positive

    def calibrate(self, positive_scores: Iterable[float]) -> SensitivityCertificate:
        scores = np.asarray(tuple(positive_scores), dtype=float)
        if scores.ndim != 1:
            raise ValueError("positive_scores must be a one-dimensional sequence.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("positive_scores must contain only finite values.")

        n_positive = int(scores.size)
        k_star = self.allowed_misses(n_positive)
        if k_star is None:
            return SensitivityCertificate(
                status=CertificateStatus.REFUSE,
                threshold=None,
                n_positive=n_positive,
                allowed_misses=None,
                target_fnr=self.target_fnr,
                failure_probability=self.failure_probability,
                reason=(
                    "insufficient_positive_calibration_cases: "
                    f"need at least {self.minimum_positive_count}, got {n_positive}"
                ),
            )

        threshold = float(np.sort(scores, kind="mergesort")[k_star])
        return SensitivityCertificate(
            status=CertificateStatus.CERTIFIED,
            threshold=threshold,
            n_positive=n_positive,
            allowed_misses=k_star,
            target_fnr=self.target_fnr,
            failure_probability=self.failure_probability,
        )
