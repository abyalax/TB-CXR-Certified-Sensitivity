"""Deterministic checks for the P0.7 risk-control baselines."""

from __future__ import annotations

import numpy as np
import pytest

from tb_certify.calibration.risk_control import (
    ImportanceWeightedLTTCalibrator,
    MarginalCRCCalibrator,
    RiskControlStatus,
    WeightedCRCCalibrator,
)


def test_marginal_crc_uses_n_plus_one_correction() -> None:
    certificate = MarginalCRCCalibrator(0.10).calibrate(np.arange(10, dtype=float))

    assert certificate.status is RiskControlStatus.ACCEPTED
    assert certificate.threshold == 0.0
    assert certificate.corrected_empirical_risk == pytest.approx(1 / 11)


def test_marginal_crc_refuses_without_a_finite_corrected_threshold() -> None:
    certificate = MarginalCRCCalibrator(0.10).calibrate([0.1] * 8)

    assert certificate.status is RiskControlStatus.REFUSE
    assert certificate.threshold is None


def test_w_crc_uses_each_test_weight_in_augmented_loss() -> None:
    certificates = WeightedCRCCalibrator(0.10).calibrate(
        calibration_scores=[0.1, 0.2, 0.3],
        calibration_weights=[3.0, 3.0, 3.0],
        test_weights=[1.0, 2.0],
    )

    assert certificates[0].status is RiskControlStatus.ACCEPTED
    assert certificates[0].threshold == 0.1
    assert certificates[0].corrected_empirical_risk == pytest.approx(0.1)
    assert certificates[1].status is RiskControlStatus.REFUSE
    assert certificates[1].threshold is None


def test_weighted_ltt_selects_safe_prefix_with_wsr_fixed_sequence() -> None:
    certificate = ImportanceWeightedLTTCalibrator(
        target_fnr=0.10,
        failure_probability=0.05,
        permutations=7,
        seed=7,
    ).calibrate(
        calibration_scores=np.arange(100, dtype=float),
        calibration_weights=np.ones(100),
        weight_upper_bound=1.0,
        thresholds=np.arange(1, 100, dtype=float),
    )

    assert certificate.status is RiskControlStatus.ACCEPTED
    assert certificate.threshold is not None
    assert certificate.threshold >= 0.0
    assert certificate.p_value is not None
    assert certificate.p_value <= 0.05


def test_weighted_ltt_rejects_invalid_weight_bound() -> None:
    calibrator = ImportanceWeightedLTTCalibrator(permutations=1)
    with pytest.raises(ValueError, match="bound every"):
        calibrator.calibrate(
            [0.1, 0.2],
            [1.0, 2.0],
            weight_upper_bound=1.0,
            thresholds=[0.15],
        )
