"""Tests for leakage-free selective-verification simulation."""

from __future__ import annotations

import numpy as np

from tb_certify.evaluation.selective_study import SelectiveVerificationStudy


def test_selective_study_reports_naive_and_verification_aware_methods() -> None:
    scores = np.linspace(0.01, 0.99, 80)
    labels = np.zeros(80, dtype=int)
    labels[::2] = 1
    result = SelectiveVerificationStudy().run(
        calibration_scores=scores,
        calibration_labels=labels,
        evaluation_scores=scores,
        evaluation_labels=labels,
        thresholds=[0.01, 0.05, 0.10],
        reference_threshold=0.50,
        low_score_probabilities=[0.5, 1.0],
        repetitions=3,
        seed=9,
        split_unit="image-level synthetic test",
    )

    assert len(result.summaries) == 2
    for summary in result.summaries:
        assert summary.naive_tier_a.issued + summary.naive_tier_a.refused == 3
        assert (
            summary.verification_bias_certificate.issued
            + summary.verification_bias_certificate.refused
            == 3
        )
