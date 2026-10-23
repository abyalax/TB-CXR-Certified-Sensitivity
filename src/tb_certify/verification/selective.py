"""Initial Clopper-Pearson certificate under known selective verification.

This is the conservative four-bound construction in the research notes. The
fixed threshold grid must be chosen before confirmation labels are inspected.
The tighter linear-functional certificate is a later P1 investigation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class ThresholdBound:
    threshold: float
    numerator_upper: float
    denominator_lower: float
    fnr_upper_bound: float
    passes: bool
    high_missed_positive_upper: float
    low_missed_positive_upper: float


@dataclass(frozen=True)
class SelectiveThresholdCertificate:
    threshold: float | None
    issued: bool
    target_fnr: float
    failure_probability: float
    screening_count: int
    verified_count: int
    verification_rate: float
    bounds: tuple[ThresholdBound, ...]
    refusal_reason: str | None = None


@dataclass(frozen=True)
class LinearThresholdBound:
    """One-sided IPW linear-functional bound for one fixed threshold."""

    threshold: float
    empirical_mean: float
    upper_confidence_bound: float
    passes: bool


@dataclass(frozen=True)
class LinearSelectiveThresholdCertificate:
    """Fixed-sequence certificate based on an IPW linear functional."""

    threshold: float | None
    issued: bool
    target_fnr: float
    failure_probability: float
    screening_count: int
    verified_count: int
    verification_rate: float
    bounds: tuple[LinearThresholdBound, ...]
    refusal_reason: str | None = None


class SelectiveVerificationPolicy:
    """Implement high-score verification plus an independent low-score sample."""

    @staticmethod
    def sample(
        scores: Iterable[float],
        reference_threshold: float,
        low_score_probability: float,
        seed: int,
    ) -> np.ndarray:
        score_array = np.asarray(tuple(scores), dtype=float)
        if score_array.ndim != 1 or not np.all(np.isfinite(score_array)):
            raise ValueError("scores must be a finite one-dimensional sequence.")
        if not np.isfinite(reference_threshold):
            raise ValueError("reference_threshold must be finite.")
        if not 0 < low_score_probability <= 1:
            raise ValueError("low_score_probability must be in (0, 1].")

        rng = np.random.default_rng(seed)
        high_score = score_array >= reference_threshold
        verified = high_score.copy()
        verified[~high_score] = (
            rng.random(np.sum(~high_score)) < low_score_probability
        )
        return verified


class SelectiveVerificationCalibrator:
    """Certify FNR from known-probability verification of low-scoring cases."""

    def __init__(
        self,
        thresholds: Iterable[float],
        reference_threshold: float,
        low_score_probability: float,
        target_fnr: float = 0.10,
        failure_probability: float = 0.05,
    ) -> None:
        grid = np.asarray(tuple(thresholds), dtype=float)
        if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
            raise ValueError("thresholds must be a non-empty finite sequence.")
        if np.any(np.diff(grid) <= 0):
            raise ValueError(
                "thresholds must be strictly increasing and fixed a priori."
            )
        if not np.isfinite(reference_threshold):
            raise ValueError("reference_threshold must be finite.")
        if not 0 < low_score_probability <= 1:
            raise ValueError("low_score_probability must be in (0, 1].")
        if not 0 < target_fnr < 1:
            raise ValueError("target_fnr must be strictly between 0 and 1.")
        if not 0 < failure_probability < 1:
            raise ValueError("failure_probability must be strictly between 0 and 1.")

        self.thresholds = tuple(float(value) for value in grid)
        self.reference_threshold = float(reference_threshold)
        self.low_score_probability = float(low_score_probability)
        self.target_fnr = target_fnr
        self.failure_probability = failure_probability

    def calibrate(
        self,
        scores: Iterable[float],
        verified: Iterable[bool],
        verified_labels: Iterable[int | None],
    ) -> SelectiveThresholdCertificate:
        score_array = np.asarray(tuple(scores), dtype=float)
        verified_array = np.asarray(tuple(verified), dtype=bool)
        label_values = tuple(verified_labels)
        if score_array.ndim != 1 or score_array.size == 0:
            raise ValueError("scores must be a non-empty vector.")
        if not np.all(np.isfinite(score_array)):
            raise ValueError("scores must be finite.")
        if (
            verified_array.shape != score_array.shape
            or len(label_values) != len(score_array)
        ):
            raise ValueError(
                "scores, verified flags, and labels must have equal length."
            )

        labels = np.full(score_array.size, -1, dtype=int)
        for index, label in enumerate(label_values):
            if verified_array[index]:
                if label not in (0, 1):
                    raise ValueError("Every verified record must have a binary label.")
                labels[index] = int(label)
            elif label is not None:
                raise ValueError(
                    "Unverified labels must be None; hiding them prevents leakage."
                )

        high_score = score_array >= self.reference_threshold
        if np.any(high_score & ~verified_array):
            raise ValueError(
                "All records at or above the reference threshold must be verified."
            )

        n_screened = int(score_array.size)
        is_positive = labels == 1
        observed_high_positive = int(np.sum(high_score & is_positive))
        observed_low_positive = int(np.sum(~high_score & is_positive))
        eta = self.failure_probability / 4
        bounds: list[ThresholdBound] = []

        for threshold in self.thresholds:
            high_missed = int(
                np.sum(
                    is_positive
                    & high_score
                    & (score_array < threshold)
                )
            )
            low_limit = min(threshold, self.reference_threshold)
            low_missed = int(
                np.sum(
                    is_positive
                    & ~high_score
                    & (score_array < low_limit)
                )
            )
            high_upper = (
                0.0
                if threshold <= self.reference_threshold
                else self._upper(high_missed, n_screened, eta)
            )
            low_upper = self._upper(low_missed, n_screened, eta) / (
                self.low_score_probability
            )
            high_total_lower = self._lower(
                observed_high_positive, n_screened, eta
            )
            low_total_lower = self._lower(
                observed_low_positive, n_screened, eta
            ) / self.low_score_probability
            denominator_lower = high_total_lower + low_total_lower
            numerator_upper = high_upper + low_upper
            fnr_upper = (
                numerator_upper / denominator_lower
                if denominator_lower > 0
                else float("inf")
            )
            bounds.append(
                ThresholdBound(
                    threshold=threshold,
                    numerator_upper=numerator_upper,
                    denominator_lower=denominator_lower,
                    fnr_upper_bound=fnr_upper,
                    passes=fnr_upper <= self.target_fnr,
                    high_missed_positive_upper=high_upper,
                    low_missed_positive_upper=low_upper,
                )
            )

        last_passing_threshold: float | None = None
        prefix_open = True
        for bound in bounds:
            if prefix_open and bound.passes:
                last_passing_threshold = bound.threshold
            else:
                prefix_open = False

        verified_count = int(np.sum(verified_array))
        refusal_reason = (
            None
            if last_passing_threshold is not None
            else "no_threshold_passed_fixed_sequence_certificate"
        )
        return SelectiveThresholdCertificate(
            threshold=last_passing_threshold,
            issued=last_passing_threshold is not None,
            target_fnr=self.target_fnr,
            failure_probability=self.failure_probability,
            screening_count=n_screened,
            verified_count=verified_count,
            verification_rate=verified_count / n_screened,
            bounds=tuple(bounds),
            refusal_reason=refusal_reason,
        )

    @staticmethod
    def _upper(successes: int, trials: int, tail_probability: float) -> float:
        if successes == trials:
            return 1.0
        return float(beta.ppf(1 - tail_probability, successes + 1, trials - successes))

    @staticmethod
    def _lower(successes: int, trials: int, tail_probability: float) -> float:
        if successes == 0:
            return 0.0
        return float(beta.ppf(tail_probability, successes, trials - successes + 1))


class LinearSelectiveVerificationCalibrator:
    """A tighter one-functional alternative to the four-CP VB construction.

    For fixed threshold ``t``, the inverse-probability weighted quantity
    ``V Y (1{score < t} - alpha) / pi(score)`` has non-positive expectation
    exactly when the population FNR is at most ``alpha`` (for nonzero TB
    prevalence).  A heterogeneous-range Hoeffding UCB tests this directly,
    avoiding a separately bounded numerator and denominator.  As with the
    original construction, thresholds are a fixed, increasing sequence.
    """

    def __init__(
        self,
        thresholds: Iterable[float],
        reference_threshold: float,
        low_score_probability: float,
        target_fnr: float = 0.10,
        failure_probability: float = 0.05,
    ) -> None:
        self._delegate = SelectiveVerificationCalibrator(
            thresholds=thresholds,
            reference_threshold=reference_threshold,
            low_score_probability=low_score_probability,
            target_fnr=target_fnr,
            failure_probability=failure_probability,
        )

    def calibrate(
        self,
        scores: Iterable[float],
        verified: Iterable[bool],
        verified_labels: Iterable[int | None],
    ) -> LinearSelectiveThresholdCertificate:
        score_array = np.asarray(tuple(scores), dtype=float)
        verified_array = np.asarray(tuple(verified), dtype=bool)
        label_values = tuple(verified_labels)
        # Delegate performs all input, masking, and high-score checks.
        self._delegate.calibrate(score_array, verified_array, label_values)
        labels = np.full(score_array.size, -1, dtype=int)
        for index, label in enumerate(label_values):
            if verified_array[index]:
                assert label in (0, 1)
                labels[index] = int(label)

        probabilities = np.where(
            score_array >= self._delegate.reference_threshold,
            1.0,
            self._delegate.low_score_probability,
        )
        if not np.any((labels == 1) & verified_array):
            return LinearSelectiveThresholdCertificate(
                threshold=None,
                issued=False,
                target_fnr=self._delegate.target_fnr,
                failure_probability=self._delegate.failure_probability,
                screening_count=int(score_array.size),
                verified_count=int(np.sum(verified_array)),
                verification_rate=float(np.mean(verified_array)),
                bounds=(),
                refusal_reason="no_verified_positive_records",
            )

        range_widths = 1 / probabilities
        deviation = float(
            np.sqrt(
                np.log(1 / self._delegate.failure_probability)
                * np.sum(range_widths**2)
                / (2 * score_array.size**2)
            )
        )
        bounds: list[LinearThresholdBound] = []
        for threshold in self._delegate.thresholds:
            misses = score_array < threshold
            values = (
                verified_array
                * (labels == 1)
                * (misses.astype(float) - self._delegate.target_fnr)
                / probabilities
            )
            empirical_mean = float(np.mean(values))
            upper_bound = empirical_mean + deviation
            bounds.append(
                LinearThresholdBound(
                    threshold=threshold,
                    empirical_mean=empirical_mean,
                    upper_confidence_bound=upper_bound,
                    passes=upper_bound <= 0,
                )
            )

        selected: float | None = None
        for bound in bounds:
            if not bound.passes:
                break
            selected = bound.threshold
        return LinearSelectiveThresholdCertificate(
            threshold=selected,
            issued=selected is not None,
            target_fnr=self._delegate.target_fnr,
            failure_probability=self._delegate.failure_probability,
            screening_count=int(score_array.size),
            verified_count=int(np.sum(verified_array)),
            verification_rate=float(np.mean(verified_array)),
            bounds=tuple(bounds),
            refusal_reason=(
                None
                if selected is not None
                else "no_threshold_passed_linear_certificate"
            ),
        )
