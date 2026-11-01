"""Exact floor, staircase, refusal, and tie checks for Tier A."""

from __future__ import annotations

import numpy as np

from tb_certify.calibration.tier_a import CertificateStatus, TierACalibrator


def test_tier_a_refuses_below_minimum_positive_count() -> None:
    calibrator = TierACalibrator(target_fnr=0.10, failure_probability=0.05)

    certificate = calibrator.calibrate(np.arange(28, dtype=float))

    assert calibrator.minimum_positive_count == 29
    assert certificate.status is CertificateStatus.REFUSE
    assert certificate.threshold is None


def test_tier_a_floor_and_order_statistic_staircase() -> None:
    calibrator = TierACalibrator(target_fnr=0.10, failure_probability=0.05)

    assert [calibrator.allowed_misses(n) for n in (29, 46, 61, 76)] == [0, 1, 2, 3]


def test_tier_a_ties_at_selected_order_statistic_are_included() -> None:
    calibrator = TierACalibrator(target_fnr=0.10, failure_probability=0.05)
    scores = np.array([0.0, 1.0, 1.0, *np.arange(2, 45, dtype=float)])

    certificate = calibrator.calibrate(scores)

    assert certificate.status is CertificateStatus.CERTIFIED
    assert certificate.allowed_misses == 1
    assert certificate.threshold == 1.0
    assert np.count_nonzero(scores < certificate.threshold) == 1


def test_tier_a_synthetic_continuous_scores_reproduce_analytic_violation() -> None:
    """Test that allowed_misses matches analytic binomial violation probabilities.

    The allowed_misses k* is defined as max{k: P(Bin(n, alpha) <= k) <= delta}.
    This test verifies that the implementation matches the analytic formula.
    """
    from scipy.stats import binom

    calibrator = TierACalibrator(target_fnr=0.10, failure_probability=0.05)

    # Test a range of n values against the analytic binomial CDF
    test_cases = list(range(29, 153))  # From floor to k=8

    for n in test_cases:
        k = calibrator.allowed_misses(n)
        analytic_prob = binom.cdf(k, n, 0.10)

        # For k* defined as max{k: P(Bin(n, alpha) <= k) <= delta}:
        # - k must satisfy P(Bin(n, alpha) <= k) <= delta
        # - k+1 must satisfy P(Bin(n, alpha) <= k+1) > delta (if k+1 exists)
        assert analytic_prob <= 0.05, (
            f"n={n}, k={k}: P(Bin({n}, 0.10) <= {k}) = {analytic_prob:.4f} > 0.05"
        )

        if k + 1 <= n:
            analytic_prob_k1 = binom.cdf(k + 1, n, 0.10)
            assert analytic_prob_k1 > 0.05, (
                f"n={n}, k={k}: P(Bin({n}, 0.10) <= {k+1}) = {analytic_prob_k1:.4f} <= 0.05, "
                f"should have allowed k+1={k+1} misses"
            )
