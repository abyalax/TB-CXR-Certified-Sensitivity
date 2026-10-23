"""Repeated finite-sample audit of Tier A on a fixed target holdout."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tb_certify.calibration.tier_a import CertificateStatus, TierACalibrator
from tb_certify.evaluation.binomial import BinomialInterval, clopper_pearson_interval
from tb_certify.evaluation.metrics import evaluate_threshold


@dataclass(frozen=True)
class TierAStudySummary:
    requested_positive_count: int
    available_calibration_positive_count: int
    refusal_reason: str | None
    repetitions: int
    certified_repetitions: int
    refused_repetitions: int
    refusal_rate: float
    violations: int
    violation_rate: float | None
    violation_rate_interval: BinomialInterval | None
    mean_evaluation_sensitivity: float | None
    mean_evaluation_specificity: float | None
    probability_specificity_at_least_target: float | None
    target_specificity: float
    evaluation_positive_count: int
    evaluation_negative_count: int
    split_unit: str


@dataclass(frozen=True)
class TierAStudyResult:
    target_fnr: float
    failure_probability: float
    seed: int
    resampling: str
    summaries: tuple[TierAStudySummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TierAStudy:
    """Audit calibration draws against an independent, fully labeled holdout.

    Each repetition draws distinct positive records without replacement from
    the supplied calibration pool. Repetitions are independent conditional on
    that finite pool. The reported binomial interval describes Monte Carlo
    uncertainty in this empirical audit; the finite holdout itself remains an
    estimate of the unknown target population.
    """

    def __init__(
        self,
        calibrator: TierACalibrator | None = None,
        target_specificity: float = 0.70,
        confidence: float = 0.95,
    ) -> None:
        if not 0 < target_specificity < 1:
            raise ValueError("target_specificity must be strictly between 0 and 1.")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be strictly between 0 and 1.")
        self.calibrator = calibrator or TierACalibrator()
        self.target_specificity = target_specificity
        self.confidence = confidence

    def run(
        self,
        calibration_positive_scores: Iterable[float],
        evaluation_scores: Iterable[float],
        evaluation_labels: Iterable[int],
        calibration_sizes: Sequence[int],
        repetitions: int = 1000,
        seed: int = 42,
        split_unit: str = "unspecified",
        calibration_group_ids: Sequence[str] | None = None,
        evaluation_group_ids: Sequence[str] | None = None,
    ) -> TierAStudyResult:
        calibration_pool = np.asarray(tuple(calibration_positive_scores), dtype=float)
        test_scores = np.asarray(tuple(evaluation_scores), dtype=float)
        test_labels = np.asarray(tuple(evaluation_labels), dtype=int)
        if calibration_pool.ndim != 1 or calibration_pool.size == 0:
            raise ValueError("calibration_positive_scores must be a non-empty vector.")
        if not np.all(np.isfinite(calibration_pool)):
            raise ValueError("Calibration scores must be finite.")
        if test_scores.ndim != 1 or test_labels.ndim != 1:
            raise ValueError("Evaluation scores and labels must be vectors.")
        if test_scores.size != test_labels.size or test_scores.size == 0:
            raise ValueError("Evaluation scores and labels must have equal non-zero size.")  # noqa: E501
        if not np.all(np.isfinite(test_scores)):
            raise ValueError("Evaluation scores must be finite.")
        if not np.isin(test_labels, [0, 1]).all():
            raise ValueError("Evaluation labels must use 1 for active TB and 0 for non-TB.")  # noqa: E501
        if repetitions < 1:
            raise ValueError("repetitions must be at least one.")
        if len(calibration_sizes) == 0 or any(
            not isinstance(size, (int, np.integer)) or size < 1
            for size in calibration_sizes
        ):
            raise ValueError("calibration_sizes must contain positive integers.")
        if len(set(calibration_sizes)) != len(calibration_sizes):
            raise ValueError("calibration_sizes must not contain duplicates.")

        if (calibration_group_ids is None) != (evaluation_group_ids is None):
            raise ValueError("Provide group IDs for both pools or neither pool.")
        if split_unit == "patient" and calibration_group_ids is None:
            raise ValueError("Patient-level studies require group IDs for both pools.")
        self._validate_group_ids(
            calibration_group_ids, len(calibration_pool), "calibration_group_ids"
        )
        self._validate_group_ids(
            evaluation_group_ids, len(test_scores), "evaluation_group_ids"
        )
        if calibration_group_ids is not None and evaluation_group_ids is not None:
            overlap = set(calibration_group_ids).intersection(evaluation_group_ids)
            if overlap:
                raise ValueError(
                    f"Calibration and evaluation records overlap ({len(overlap)} IDs)."
                )

        rng = np.random.default_rng(seed)
        summaries: list[TierAStudySummary] = []
        evaluation_positive_count = int(np.sum(test_labels == 1))
        evaluation_negative_count = int(np.sum(test_labels == 0))
        if not evaluation_positive_count or not evaluation_negative_count:
            raise ValueError("The evaluation holdout must contain both classes.")

        for sample_size in calibration_sizes:
            certified_metrics = []
            violations = 0
            refused_repetitions = 0
            if sample_size > calibration_pool.size:
                refused_repetitions = repetitions
            else:
                for _ in range(repetitions):
                    draw = rng.choice(calibration_pool, size=sample_size, replace=False)
                    certificate = self.calibrator.calibrate(draw)
                    if certificate.status is CertificateStatus.REFUSE:
                        refused_repetitions += 1
                        continue
                    assert certificate.threshold is not None
                    metrics = evaluate_threshold(
                        test_scores, test_labels, certificate.threshold
                    )
                    assert metrics.sensitivity is not None
                    assert metrics.specificity is not None
                    certified_metrics.append(metrics)
                    if 1 - metrics.sensitivity > self.calibrator.target_fnr:
                        violations += 1

            n_certified = len(certified_metrics)
            interval = (
                clopper_pearson_interval(
                    violations, n_certified, confidence=self.confidence
                )
                if n_certified
                else None
            )
            summaries.append(
                TierAStudySummary(
                    requested_positive_count=sample_size,
                    available_calibration_positive_count=int(calibration_pool.size),
                    refusal_reason=(
                        "requested_size_exceeds_available_pool"
                        if sample_size > calibration_pool.size
                        else (
                            "below_exact_certification_floor"
                            if self.calibrator.allowed_misses(sample_size) is None
                            else None
                        )
                    ),
                    repetitions=repetitions,
                    certified_repetitions=n_certified,
                    refused_repetitions=refused_repetitions,
                    refusal_rate=refused_repetitions / repetitions,
                    violations=violations,
                    violation_rate=violations / n_certified if n_certified else None,
                    violation_rate_interval=interval,
                    mean_evaluation_sensitivity=(
                        float(np.mean([row.sensitivity for row in certified_metrics]))
                        if certified_metrics
                        else None
                    ),
                    mean_evaluation_specificity=(
                        float(np.mean([row.specificity for row in certified_metrics]))
                        if certified_metrics
                        else None
                    ),
                    probability_specificity_at_least_target=(
                        float(
                            np.mean(
                                [
                                    row.specificity >= self.target_specificity
                                    for row in certified_metrics
                                ]
                            )
                        )
                        if certified_metrics
                        else None
                    ),
                    target_specificity=self.target_specificity,
                    evaluation_positive_count=evaluation_positive_count,
                    evaluation_negative_count=evaluation_negative_count,
                    split_unit=split_unit,
                )
            )

        return TierAStudyResult(
            target_fnr=self.calibrator.target_fnr,
            failure_probability=self.calibrator.failure_probability,
            seed=seed,
            resampling="independent simple random samples without replacement",
            summaries=tuple(summaries),
        )

    @staticmethod
    def _validate_group_ids(
        identifiers: Sequence[str] | None, expected_length: int, name: str
    ) -> None:
        if identifiers is None:
            return
        if len(identifiers) != expected_length:
            raise ValueError(f"{name} length must match its score array.")
        if any(not identifier for identifier in identifiers):
            raise ValueError(f"{name} must not contain empty identifiers.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(
                f"{name} must identify one score per independent study unit. "
                "Aggregate multiple images per patient before running the study."
            )
