"""Unit checks for selective-verification certificates."""

from __future__ import annotations

import numpy as np

from tb_certify.verification.selective import (
    LinearSelectiveVerificationCalibrator,
    SelectiveVerificationPolicy,
)


def test_linear_certificate_uses_masked_labels_and_fixed_prefix() -> None:
    scores = np.linspace(0.01, 0.99, 100)
    labels = np.zeros(100, dtype=int)
    labels[::4] = 1
    verified = SelectiveVerificationPolicy.sample(scores, 0.50, 1.0, seed=2)
    masked = tuple(
        int(label) if is_verified else None
        for label, is_verified in zip(labels, verified, strict=True)
    )
    certificate = LinearSelectiveVerificationCalibrator(
        thresholds=[0.01, 0.05, 0.10],
        reference_threshold=0.50,
        low_score_probability=1.0,
    ).calibrate(scores, verified, masked)

    assert certificate.screening_count == 100
    assert certificate.verified_count == 100
    assert len(certificate.bounds) == 3
    assert all(
        left.threshold < right.threshold
        for left, right in zip(
            certificate.bounds, certificate.bounds[1:], strict=False
        )
    )
