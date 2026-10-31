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
