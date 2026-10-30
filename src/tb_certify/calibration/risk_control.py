"""Sensitivity-risk baselines under exchangeability and covariate shift.

All routines use the TB-positive miss loss ``1{score < threshold}``, where a
higher score means stronger evidence for active TB.  They are intentionally
separate from :mod:`tier_a`: CRC controls *marginal* expected risk, whereas
the LTT routine controls risk with high probability over its calibration set.

The weighted procedures require a pre-specified, valid likelihood ratio
``d P_target(X | Y=1) / d P_source(X | Y=1)``.  Estimating those ratios on the
same data, clipping them, or applying them when conditional covariate shift is
not credible invalidates the stated guarantees.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np


class RiskControlStatus(str, Enum):
    """Whether the requested rule was found in the supplied threshold grid."""

    ACCEPTED = "accepted"
    REFUSE = "refuse"


@dataclass(frozen=True)
class MarginalCRCCertificate:
    """Output of marginal conformal risk control (CRC)."""

    status: RiskControlStatus
    threshold: float | None
    n_calibration: int
    target_fnr: float
    corrected_empirical_risk: float | None
    reason: str | None = None


@dataclass(frozen=True)
class WeightedCRCCertificate:
    """W-CRC output for one deployment covariate weight.

    W-CRC is a marginal-risk method, not a ``1-delta`` PAC certificate.  The
    selected threshold can differ from one test covariate to the next.
    """

    status: RiskControlStatus
    threshold: float | None
    test_weight: float
    calibration_weight_sum: float
    target_fnr: float
    corrected_empirical_risk: float | None
    reason: str | None = None


@dataclass(frozen=True)
class WeightedLTTCertificate:
    """High-probability weighted-LTT selection result."""

    status: RiskControlStatus
    threshold: float | None
    target_fnr: float
    failure_probability: float
    p_value: float | None
    tested_thresholds: int
    weight_upper_bound: float
    permutations: int
    reason: str | None = None


def _scores(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _weights(values: Iterable[float], expected_size: int) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.ndim != 1 or array.size != expected_size:
        raise ValueError("calibration_weights must match calibration_scores.")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("calibration_weights must be finite and non-negative.")
    if not np.any(array > 0):
        raise ValueError("calibration_weights must contain a positive value.")
    return array


def _probability(value: float, name: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between zero and one.")


def _threshold_grid(scores: np.ndarray) -> np.ndarray:
    """Finite thresholds, ordered from safest to least sensitive."""

    return np.unique(scores)


def _fixed_threshold_grid(values: Iterable[float]) -> np.ndarray:
    """Validate a candidate sequence fixed independently of LTT calibration data."""
    thresholds = np.asarray(tuple(values), dtype=float)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(thresholds)):
        raise ValueError("thresholds must contain only finite values.")
    if np.any(np.diff(thresholds) <= 0):
        raise ValueError("thresholds must be strictly increasing and unique.")
    return thresholds


class MarginalCRCCalibrator:
    """CRC for expected target FNR under positive-class exchangeability.

    This is the finite-sample correction from conformal risk control: select
    the greatest threshold whose ``(sum(loss) + 1) / (n + 1)`` is at most the
    requested FNR.  It does not provide a high-probability/PAC guarantee.
    """

    def __init__(self, target_fnr: float = 0.10) -> None:
        _probability(target_fnr, "target_fnr")
        self.target_fnr = target_fnr

    def calibrate(self, positive_scores: Iterable[float]) -> MarginalCRCCertificate:
        scores = _scores(positive_scores, "positive_scores")
        thresholds = _threshold_grid(scores)
        losses = scores[:, None] < thresholds[None, :]
        corrected_risks = (np.sum(losses, axis=0) + 1) / (scores.size + 1)
        accepted = np.flatnonzero(corrected_risks <= self.target_fnr)
        if not accepted.size:
            return MarginalCRCCertificate(
                status=RiskControlStatus.REFUSE,
                threshold=None,
                n_calibration=int(scores.size),
                target_fnr=self.target_fnr,
                corrected_empirical_risk=None,
                reason="no_finite_threshold_satisfies_crc_correction",
            )
        index = int(accepted[-1])
        return MarginalCRCCertificate(
            status=RiskControlStatus.ACCEPTED,
            threshold=float(thresholds[index]),
            n_calibration=int(scores.size),
            target_fnr=self.target_fnr,
            corrected_empirical_risk=float(corrected_risks[index]),
        )


class WeightedCRCCalibrator:
    """Test-covariate-conditional W-CRC threshold selection.

    Implements the W-CRC augmented-loss rule used by Zecchin et al.  For each
    deployment point with likelihood ratio ``w_test``, it finds the greatest
    threshold ``t`` satisfying

    ``(sum_i w_i 1{score_i < t} + w_test) / (sum_i w_i + w_test) <= alpha``.
    """

    def __init__(self, target_fnr: float = 0.10) -> None:
        _probability(target_fnr, "target_fnr")
        self.target_fnr = target_fnr

    def calibrate(
        self,
        calibration_scores: Iterable[float],
        calibration_weights: Iterable[float],
        test_weights: Iterable[float],
    ) -> tuple[WeightedCRCCertificate, ...]:
        scores = _scores(calibration_scores, "calibration_scores")
        weights = _weights(calibration_weights, scores.size)
        deployment_weights = np.asarray(tuple(test_weights), dtype=float)
        if deployment_weights.ndim != 1:
            raise ValueError("test_weights must be a one-dimensional sequence.")
        if not np.all(np.isfinite(deployment_weights)) or np.any(
            deployment_weights < 0
        ):
            raise ValueError("test_weights must be finite and non-negative.")

        thresholds = _threshold_grid(scores)
        weighted_losses = np.sum(
            weights[:, None] * (scores[:, None] < thresholds[None, :]), axis=0
        )
        total_weight = float(np.sum(weights))
        certificates: list[WeightedCRCCertificate] = []
        for test_weight in deployment_weights:
            corrected_risks = (weighted_losses + test_weight) / (
                total_weight + test_weight
            )
            accepted = np.flatnonzero(corrected_risks <= self.target_fnr)
            if not accepted.size:
                certificates.append(
                    WeightedCRCCertificate(
                        status=RiskControlStatus.REFUSE,
                        threshold=None,
                        test_weight=float(test_weight),
                        calibration_weight_sum=total_weight,
                        target_fnr=self.target_fnr,
                        corrected_empirical_risk=None,
                        reason="no_finite_threshold_satisfies_w_crc_correction",
                    )
                )
                continue
            index = int(accepted[-1])
            certificates.append(
                WeightedCRCCertificate(
                    status=RiskControlStatus.ACCEPTED,
                    threshold=float(thresholds[index]),
                    test_weight=float(test_weight),
                    calibration_weight_sum=total_weight,
                    target_fnr=self.target_fnr,
                    corrected_empirical_risk=float(corrected_risks[index]),
                )
            )
        return tuple(certificates)


class ImportanceWeightedLTTCalibrator:
    """Importance-weighted LTT using Almeida et al.'s WSR p-value.

    The source-positive calibration scores are assessed on a fixed threshold
    grid from safest to least safe.  Fixed-sequence testing stops at the first
    non-rejected null ``target FNR > alpha``.  Weighted losses are rescaled by
    a *known* upper bound on the density ratio before applying the WSR betting
    p-value.  ``permutations`` averages capital processes as recommended when
    source records might be ordered by subsource.
    """

    def __init__(
        self,
        target_fnr: float = 0.10,
        failure_probability: float = 0.05,
        permutations: int = 100,
        seed: int = 42,
    ) -> None:
        _probability(target_fnr, "target_fnr")
        _probability(failure_probability, "failure_probability")
        if not isinstance(permutations, int) or permutations < 1:
            raise ValueError("permutations must be a positive integer.")
        self.target_fnr = target_fnr
        self.failure_probability = failure_probability
        self.permutations = permutations
        self.seed = seed

    def calibrate(
        self,
        calibration_scores: Iterable[float],
        calibration_weights: Iterable[float],
        weight_upper_bound: float,
        thresholds: Iterable[float],
    ) -> WeightedLTTCertificate:
        scores = _scores(calibration_scores, "calibration_scores")
        weights = _weights(calibration_weights, scores.size)
        candidate_thresholds = _fixed_threshold_grid(thresholds)
        if not np.isfinite(weight_upper_bound) or weight_upper_bound < 1:
            raise ValueError("weight_upper_bound must be finite and at least one.")
        if np.any(weights > weight_upper_bound):
            raise ValueError("weight_upper_bound must bound every calibration weight.")

        rng = np.random.default_rng(self.seed)
        orderings = tuple(
            rng.permutation(scores.size) for _ in range(self.permutations)
        )
        selected_threshold: float | None = None
        selected_p_value: float | None = None
        tested = 0
        for threshold in candidate_thresholds:
            losses = weights * (scores < threshold) / weight_upper_bound
            p_value = self._wsr_p_value(
                losses,
                self.target_fnr / weight_upper_bound,
                self.failure_probability,
                orderings,
            )
            tested += 1
            if p_value > self.failure_probability:
                break
            selected_threshold = float(threshold)
            selected_p_value = p_value

        if selected_threshold is None:
            return WeightedLTTCertificate(
                status=RiskControlStatus.REFUSE,
                threshold=None,
                target_fnr=self.target_fnr,
                failure_probability=self.failure_probability,
                p_value=None,
                tested_thresholds=tested,
                weight_upper_bound=float(weight_upper_bound),
                permutations=self.permutations,
                reason="safest_finite_threshold_did_not_reject_ltt_null",
            )
        return WeightedLTTCertificate(
            status=RiskControlStatus.ACCEPTED,
            threshold=selected_threshold,
            target_fnr=self.target_fnr,
            failure_probability=self.failure_probability,
            p_value=selected_p_value,
            tested_thresholds=tested,
            weight_upper_bound=float(weight_upper_bound),
            permutations=self.permutations,
        )

    @staticmethod
    def _wsr_p_value(
        scaled_losses: np.ndarray,
        scaled_target_risk: float,
        betting_delta: float,
        orderings: Sequence[np.ndarray],
    ) -> float:
        """WSR p-value after Almeida et al., Theorem 8.

        The initial capital is included in the running maximum, so returned
        values are always in ``[0, 1]`` even when every betting capital falls.
        """

        n = scaled_losses.size
        log_capitals = np.empty((len(orderings), n), dtype=float)
        for row, ordering in enumerate(orderings):
            ordered_losses = scaled_losses[ordering]
            running_sum = 0.0
            variance_sum = 0.25
            log_capital = 0.0
            for index, loss in enumerate(ordered_losses, start=1):
                previous_variance = variance_sum / index
                bet_size = min(
                    1.0,
                    float(
                        np.sqrt(
                            (2 * np.log(1 / betting_delta))
                            / (n * previous_variance)
                        )
                    ),
                )
                factor = 1 - bet_size * (loss - scaled_target_risk)
                log_capital += float(np.log(factor))
                log_capitals[row, index - 1] = log_capital
                running_sum += float(loss)
                mean = (0.5 + running_sum) / (index + 1)
                variance_sum += float((loss - mean) ** 2)

        # log(mean(exp(K_b))) avoids overflow when n is large.
        maximum_log_capital = 0.0
        for column in log_capitals.T:
            column_maximum = float(np.max(column))
            log_mean = column_maximum + float(
                np.log(np.mean(np.exp(column - column_maximum)))
            )
            maximum_log_capital = max(maximum_log_capital, log_mean)
        if maximum_log_capital > 745:
            return 0.0
        return float(np.exp(-maximum_log_capital))
