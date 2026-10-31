"""Monte-Carlo audit of naive and verification-bias-aware calibration."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tb_certify.calibration.tier_a import CertificateStatus, TierACalibrator
from tb_certify.evaluation.metrics import evaluate_threshold
from tb_certify.verification.selective import (
    LinearSelectiveVerificationCalibrator,
    SelectiveVerificationCalibrator,
    SelectiveVerificationPolicy,
)


@dataclass(frozen=True)
class SelectiveMethodSummary:
    issued: int
    refused: int
    issue_rate: float
    violations: int
    violation_rate: float | None
    mean_sensitivity: float | None
    mean_specificity: float | None


@dataclass(frozen=True)
class SelectiveStudySummary:
    low_score_probability: float
    reference_threshold: float
    naive_tier_a: SelectiveMethodSummary
    verification_bias_certificate: SelectiveMethodSummary
    linear_verification_bias_certificate: SelectiveMethodSummary


@dataclass(frozen=True)
class SelectiveStudyResult:
    target_fnr: float
    failure_probability: float
    repetitions: int
    seed: int
    calibration_count: int
    evaluation_count: int
    threshold_grid: tuple[float, ...]
    split_unit: str
    summaries: tuple[SelectiveStudySummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SelectiveVerificationStudy:
    """Audit fixed selective-verification policies on fully labelled data.

    The calibration labels are intentionally hidden from each method unless its
    simulated verification indicator is true.  Evaluation labels are held out
    and used only to quantify empirical violation.  Results are an image-level
    audit unless callers provide one independent score per patient.
    """

    def __init__(
        self,
        target_fnr: float = 0.10,
        failure_probability: float = 0.05,
    ) -> None:
        self.naive_calibrator = TierACalibrator(
            target_fnr=target_fnr, failure_probability=failure_probability
        )
        self.target_fnr = target_fnr
        self.failure_probability = failure_probability

    def run(
        self,
        calibration_scores: Iterable[float],
        calibration_labels: Iterable[int],
        evaluation_scores: Iterable[float],
        evaluation_labels: Iterable[int],
        thresholds: Sequence[float],
        reference_threshold: float,
        low_score_probabilities: Sequence[float],
        repetitions: int = 1000,
        seed: int = 42,
        split_unit: str = "unspecified",
    ) -> SelectiveStudyResult:
        cal_scores, cal_labels = self._inputs(
            calibration_scores, calibration_labels, "calibration"
        )
        eval_scores, eval_labels = self._inputs(
            evaluation_scores, evaluation_labels, "evaluation"
        )
        if not np.any(cal_labels == 1):
            raise ValueError("Calibration pool must contain at least one positive.")
        if not np.any(eval_labels == 1) or not np.any(eval_labels == 0):
            raise ValueError("Evaluation pool must contain both classes.")
        grid = tuple(float(value) for value in thresholds)
        if not grid:
            raise ValueError("thresholds must be non-empty.")
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if not low_score_probabilities:
            raise ValueError("low_score_probabilities must be non-empty.")
        if any(not 0 < probability <= 1 for probability in low_score_probabilities):
            raise ValueError("low_score_probabilities must lie in (0, 1].")

        rng = np.random.default_rng(seed)
        summaries: list[SelectiveStudySummary] = []
        for probability in low_score_probabilities:
            naive_rows: list[tuple[float, float]] = []
            vb_rows: list[tuple[float, float]] = []
            linear_rows: list[tuple[float, float]] = []
            naive_violations = 0
            vb_violations = 0
            linear_violations = 0
            for _ in range(repetitions):
                policy_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                verified = SelectiveVerificationPolicy.sample(
                    cal_scores,
                    reference_threshold=reference_threshold,
                    low_score_probability=float(probability),
                    seed=policy_seed,
                )
                observed_positive_scores = cal_scores[verified & (cal_labels == 1)]
                naive = self.naive_calibrator.calibrate(observed_positive_scores)
                if naive.status is CertificateStatus.CERTIFIED:
                    assert naive.threshold is not None
                    metrics = evaluate_threshold(
                        eval_scores, eval_labels, naive.threshold
                    )
                    assert metrics.sensitivity is not None
                    assert metrics.specificity is not None
                    naive_rows.append((metrics.sensitivity, metrics.specificity))
                    naive_violations += int(1 - metrics.sensitivity > self.target_fnr)

                hidden_labels: tuple[int | None, ...] = tuple(
                    int(label) if is_verified else None
                    for label, is_verified in zip(cal_labels, verified, strict=True)
                )
                vb = SelectiveVerificationCalibrator(
                    thresholds=grid,
                    reference_threshold=reference_threshold,
                    low_score_probability=float(probability),
                    target_fnr=self.target_fnr,
                    failure_probability=self.failure_probability,
                ).calibrate(cal_scores, verified, hidden_labels)
                if vb.issued:
                    assert vb.threshold is not None
                    metrics = evaluate_threshold(eval_scores, eval_labels, vb.threshold)
                    assert metrics.sensitivity is not None
                    assert metrics.specificity is not None
                    vb_rows.append((metrics.sensitivity, metrics.specificity))
                    vb_violations += int(1 - metrics.sensitivity > self.target_fnr)
                linear = LinearSelectiveVerificationCalibrator(
                    thresholds=grid,
                    reference_threshold=reference_threshold,
                    low_score_probability=float(probability),
                    target_fnr=self.target_fnr,
                    failure_probability=self.failure_probability,
                ).calibrate(cal_scores, verified, hidden_labels)
                if linear.issued:
                    assert linear.threshold is not None
                    metrics = evaluate_threshold(
                        eval_scores, eval_labels, linear.threshold
                    )
                    assert metrics.sensitivity is not None
                    assert metrics.specificity is not None
                    linear_rows.append((metrics.sensitivity, metrics.specificity))
                    linear_violations += int(1 - metrics.sensitivity > self.target_fnr)

            summaries.append(
                SelectiveStudySummary(
                    low_score_probability=float(probability),
                    reference_threshold=float(reference_threshold),
                    naive_tier_a=self._summary(
                        naive_rows, naive_violations, repetitions
                    ),
                    verification_bias_certificate=self._summary(
                        vb_rows, vb_violations, repetitions
                    ),
                    linear_verification_bias_certificate=self._summary(
                        linear_rows, linear_violations, repetitions
                    ),
                )
            )
        return SelectiveStudyResult(
            target_fnr=self.target_fnr,
            failure_probability=self.failure_probability,
            repetitions=repetitions,
            seed=seed,
            calibration_count=int(cal_scores.size),
            evaluation_count=int(eval_scores.size),
            threshold_grid=grid,
            split_unit=split_unit,
            summaries=tuple(summaries),
        )

    @staticmethod
    def _inputs(
        scores: Iterable[float], labels: Iterable[int], name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        score_array = np.asarray(tuple(scores), dtype=float)
        label_array = np.asarray(tuple(labels), dtype=int)
        if score_array.ndim != 1 or label_array.ndim != 1:
            raise ValueError(f"{name} scores and labels must be vectors.")
        if score_array.size == 0 or score_array.size != label_array.size:
            raise ValueError(f"{name} scores and labels must have equal non-zero size.")
        if not np.all(np.isfinite(score_array)):
            raise ValueError(f"{name} scores must be finite.")
        if not np.isin(label_array, [0, 1]).all():
            raise ValueError(f"{name} labels must be binary.")
        return score_array, label_array

    @staticmethod
    def _summary(
        rows: list[tuple[float, float]], violations: int, repetitions: int
    ) -> SelectiveMethodSummary:
        issued = len(rows)
        return SelectiveMethodSummary(
            issued=issued,
            refused=repetitions - issued,
            issue_rate=issued / repetitions,
            violations=violations,
            violation_rate=violations / issued if issued else None,
            mean_sensitivity=float(np.mean([row[0] for row in rows])) if rows else None,
            mean_specificity=float(np.mean([row[1] for row in rows])) if rows else None,
        )
