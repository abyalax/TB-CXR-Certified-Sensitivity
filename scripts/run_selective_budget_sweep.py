"""Bootstrap a selective-verification budget grid from TBX11K score pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scripts.run_primary_real_study import (
    PRIMARY_SOURCE_SITES,
    _fit_head,
    _read_csv,
    _scores,
    _source_split,
)

from tb_certify.calibration.tier_a import CertificateStatus, TierACalibrator
from tb_certify.data.identity_audit import patient_level_supported
from tb_certify.evaluation.binomial import clopper_pearson_interval
from tb_certify.evaluation.metrics import evaluate_threshold
from tb_certify.verification.selective import (
    LinearSelectiveVerificationCalibrator,
    SelectiveVerificationCalibrator,
    SelectiveVerificationPolicy,
)

SCREENING_BUDGETS = (1000, 2000, 4000, 6496)
PREVALENCES = (0.05, 0.10, 0.20, 0.40)
LOW_SCORE_PROBABILITIES = (0.10, 0.25, 0.50, 1.00)
TARGET_FNR = 0.10
FAILURE_PROBABILITY = 0.05


def _summary(
    issued: int,
    violations: int,
    sensitivities: list[float],
    specificities: list[float],
    verified_counts: list[int],
    repetitions: int,
    screening_budget: int,
) -> dict[str, Any]:
    return {
        "issued": issued,
        "refused": repetitions - issued,
        "issue_rate": issued / repetitions,
        "violations": violations,
        "violation_rate": violations / issued if issued else None,
        "violation_rate_cp95": (
            clopper_pearson_interval(violations, issued).__dict__ if issued else None
        ),
        "mean_evaluation_sensitivity": (
            float(np.mean(sensitivities)) if sensitivities else None
        ),
        "mean_evaluation_specificity": (
            float(np.mean(specificities)) if specificities else None
        ),
        "mean_verified_count": (
            float(np.mean(verified_counts)) if verified_counts else None
        ),
        "mean_verification_rate": (
            float(np.mean(verified_counts)) / screening_budget
            if verified_counts and screening_budget
            else None
        ),
    }


def _evaluate_issued(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    target_fnr: float,
) -> tuple[float, float, int]:
    metrics = evaluate_threshold(scores, labels, threshold)
    assert metrics.sensitivity is not None
    assert metrics.specificity is not None
    return (
        metrics.sensitivity,
        metrics.specificity,
        int(1 - metrics.sensitivity > target_fnr),
    )


def run_sweep(
    project_root: Path,
    embedding_dir: Path,
    output_dir: Path,
    repetitions: int,
    seed: int,
    evaluation_count: int,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    embedding_dir = embedding_dir.resolve()
    output_dir = output_dir.resolve()
    rows = _read_csv(embedding_dir / "manifest.csv")
    features = np.load(embedding_dir / "embeddings.npy", mmap_mode="r")
    if features.ndim != 2 or features.shape[0] != len(rows):
        raise ValueError("Embedding rows must align one-to-one with the manifest.")

    train, validation, _ = _source_split(rows, PRIMARY_SOURCE_SITES, seed)
    head_name = "logistic"
    head, source_validation_scores, _, reference_threshold = _fit_head(
        head_name,
        features,
        rows,
        train,
        validation,
        "TBX11K",
        PRIMARY_SOURCE_SITES,
        seed,
    )
    identity_audit = json.loads(
        (project_root / "artifacts" / "audits" / "image_identity.json").read_text(
            encoding="utf-8"
        )
    )
    duplicate_aliases = {
        (site, image_id)
        for site, site_audit in identity_audit[
            "exact_decoded_rgb_pixel_duplicates"
        ].items()
        for group in site_audit["duplicate_groups"]
        for image_id in group[1:]
    }
    target_train = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["site"] == "TBX11K"
            and row["source_split"] == "train"
            and (row["site"], row["image_id"]) not in duplicate_aliases
        ],
        dtype=int,
    )
    target_labels = np.asarray(
        [int(rows[index]["label"] == "active_tb") for index in target_train],
        dtype=int,
    )
    target_scores = _scores(head, head_name, features[target_train])
    positive_scores = target_scores[target_labels == 1]
    negative_scores = target_scores[target_labels == 0]
    threshold_grid = np.unique(
        np.quantile(source_validation_scores, np.linspace(0.01, 0.99, 25))
    )
    if not threshold_grid.size:
        raise ValueError("Source scores cannot define a fixed candidate grid.")

    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {}
    for budget in SCREENING_BUDGETS:
        results[str(budget)] = {}
        for prevalence in PREVALENCES:
            calibration_positive_count = round(budget * prevalence)
            evaluation_positive_count = round(evaluation_count * prevalence)
            results[str(budget)][str(prevalence)] = {}
            counts = {
                probability: {
                    method: {
                        "issued": 0,
                        "violations": 0,
                        "sensitivities": [],
                        "specificities": [],
                        "verified_counts": [],
                    }
                    for method in ("naive_tier_a", "four_bound_vb", "linear_ipw")
                }
                for probability in LOW_SCORE_PROBABILITIES
            }
            for _ in range(repetitions):
                calibration_scores = np.concatenate(
                    (
                        rng.choice(
                            positive_scores,
                            size=calibration_positive_count,
                            replace=True,
                        ),
                        rng.choice(
                            negative_scores,
                            size=budget - calibration_positive_count,
                            replace=True,
                        ),
                    )
                )
                calibration_labels = np.concatenate(
                    (
                        np.ones(calibration_positive_count, dtype=int),
                        np.zeros(budget - calibration_positive_count, dtype=int),
                    )
                )
                order = rng.permutation(budget)
                calibration_scores = calibration_scores[order]
                calibration_labels = calibration_labels[order]
                evaluation_scores = np.concatenate(
                    (
                        rng.choice(
                            positive_scores,
                            size=evaluation_positive_count,
                            replace=True,
                        ),
                        rng.choice(
                            negative_scores,
                            size=evaluation_count - evaluation_positive_count,
                            replace=True,
                        ),
                    )
                )
                evaluation_labels = np.concatenate(
                    (
                        np.ones(evaluation_positive_count, dtype=int),
                        np.zeros(
                            evaluation_count - evaluation_positive_count, dtype=int
                        ),
                    )
                )
                evaluation_order = rng.permutation(evaluation_count)
                evaluation_scores = evaluation_scores[evaluation_order]
                evaluation_labels = evaluation_labels[evaluation_order]

                for probability in LOW_SCORE_PROBABILITIES:
                    verified = SelectiveVerificationPolicy.sample(
                        calibration_scores,
                        reference_threshold=reference_threshold,
                        low_score_probability=probability,
                        seed=int(rng.integers(0, np.iinfo(np.int64).max)),
                    )
                    verified_count = int(np.sum(verified))
                    for method in ("naive_tier_a", "four_bound_vb", "linear_ipw"):
                        counts[probability][method]["verified_counts"].append(
                            verified_count
                        )
                    observed_positive_scores = calibration_scores[
                        verified & (calibration_labels == 1)
                    ]
                    naive = TierACalibrator(TARGET_FNR, FAILURE_PROBABILITY).calibrate(
                        observed_positive_scores
                    )
                    if naive.status is CertificateStatus.CERTIFIED:
                        assert naive.threshold is not None
                        sensitivity, specificity, violation = _evaluate_issued(
                            evaluation_scores,
                            evaluation_labels,
                            naive.threshold,
                            TARGET_FNR,
                        )
                        row = counts[probability]["naive_tier_a"]
                        row["issued"] += 1
                        row["violations"] += violation
                        row["sensitivities"].append(sensitivity)
                        row["specificities"].append(specificity)

                    hidden_labels = tuple(
                        int(label) if is_verified else None
                        for label, is_verified in zip(
                            calibration_labels, verified, strict=True
                        )
                    )
                    vb = SelectiveVerificationCalibrator(
                        thresholds=threshold_grid,
                        reference_threshold=reference_threshold,
                        low_score_probability=probability,
                        target_fnr=TARGET_FNR,
                        failure_probability=FAILURE_PROBABILITY,
                    ).calibrate(calibration_scores, verified, hidden_labels)
                    linear = LinearSelectiveVerificationCalibrator(
                        thresholds=threshold_grid,
                        reference_threshold=reference_threshold,
                        low_score_probability=probability,
                        target_fnr=TARGET_FNR,
                        failure_probability=FAILURE_PROBABILITY,
                    ).calibrate(calibration_scores, verified, hidden_labels)
                    for method, certificate in (
                        ("four_bound_vb", vb),
                        ("linear_ipw", linear),
                    ):
                        if not certificate.issued:
                            continue
                        assert certificate.threshold is not None
                        sensitivity, specificity, violation = _evaluate_issued(
                            evaluation_scores,
                            evaluation_labels,
                            certificate.threshold,
                            TARGET_FNR,
                        )
                        row = counts[probability][method]
                        row["issued"] += 1
                        row["violations"] += violation
                        row["sensitivities"].append(sensitivity)
                        row["specificities"].append(specificity)

            for probability in LOW_SCORE_PROBABILITIES:
                results[str(budget)][str(prevalence)][str(probability)] = {
                    method: _summary(
                        values["issued"],
                        values["violations"],
                        values["sensitivities"],
                        values["specificities"],
                        values["verified_counts"],
                        repetitions,
                        budget,
                    )
                    for method, values in counts[probability].items()
                }

    result = {
        "study": "selective verification budget and prevalence bootstrap sweep",
        "encoder": embedding_dir.name,
        "head": head_name,
        "reference_threshold": float(reference_threshold),
        "target_training_pool_after_exact_pixel_alias_exclusion": int(
            target_train.size
        ),
        "target_training_positive_count": int(np.sum(target_labels == 1)),
        "target_training_negative_count": int(np.sum(target_labels == 0)),
        "exact_pixel_duplicate_alias_count_excluded": len(duplicate_aliases),
        "candidate_thresholds": [float(value) for value in threshold_grid],
        "screening_budgets": list(SCREENING_BUDGETS),
        "prevalences": list(PREVALENCES),
        "low_score_verification_probabilities": list(LOW_SCORE_PROBABILITIES),
        "evaluation_count_per_repetition": evaluation_count,
        "repetitions_per_cell": repetitions,
        "seed": seed,
        "calibration_sampling": (
            "Stratified empirical bootstrap with replacement from TBX11K official "
            "training-split positive and negative score pools; prevalence is set by "
            "the grid."
        ),
        "evaluation_sampling": (
            "Independent stratified empirical bootstrap from the same "
            "class-conditional "
            "score pools; this is a simulation, not an external validation cohort."
        ),
        "verification_policy": (
            "All scores at or above the source reference threshold are verified; "
            "lower-scoring cases are independently verified at the specified q."
        ),
        "methods": [
            "naive_tier_a",
            "four_bound_vb",
            "linear_ipw",
        ],
        "results": results,
        "split_unit": (
            "image-level empirical score bootstrap; patient linkage not verified"
        ),
        "patient_level_claim": patient_level_supported(project_root),
        "pac_claim": False,
        "interpretation_limit": (
            "This budget sweep characterizes a specified empirical bootstrap model. "
            "It does not establish performance, label quality, patient independence, "
            "or a PAC guarantee for a future clinical population."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    result["output_path"] = str(output_path)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--evaluation-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if arguments.repetitions < 1 or arguments.evaluation_count < 1:
        parser.error("--repetitions and --evaluation-count must be positive")
    result = run_sweep(
        arguments.project_root,
        arguments.embedding_dir,
        arguments.output_dir,
        arguments.repetitions,
        arguments.seed,
        arguments.evaluation_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
