"""Run multi-seed, nested calibration, shift-diagnostic, and site-control audits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binom
from scripts.run_primary_real_study import (
    PRIMARY_SOURCE_SITES,
    SECONDARY_SOURCE_SITES,
    _fit_head,
    _read_csv,
    _scores,
    _source_split,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tb_certify.calibration.risk_control import (
    ImportanceWeightedLTTCalibrator,
    MarginalCRCCalibrator,
    RiskControlStatus,
)
from tb_certify.calibration.tier_a import CertificateStatus, TierACalibrator
from tb_certify.data.identity_audit import patient_level_supported
from tb_certify.evaluation.binomial import clopper_pearson_interval
from tb_certify.evaluation.metrics import evaluate_threshold

CALIBRATION_SIZES = (10, 20, 28, *range(29, 46), 46, 61, 76, 100)
SEEDS = tuple(range(42, 52))
TARGET_FNR = 0.10
FAILURE_PROBABILITY = 0.05


def _exact_duplicate_aliases(project_root: Path) -> set[tuple[str, str]]:
    audit_path = project_root / "artifacts" / "audits" / "image_identity.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    aliases: set[tuple[str, str]] = set()
    for site, site_audit in audit["exact_decoded_rgb_pixel_duplicates"].items():
        for group in site_audit["duplicate_groups"]:
            aliases.update((site, image_id) for image_id in group[1:])
    return aliases


def _target_split(
    rows: list[dict[str, str]], target_site: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(
        [index for index, row in enumerate(rows) if row["site"] == target_site],
        dtype=int,
    )
    if target_site == "TBX11K":
        train = np.asarray(
            [index for index in target if rows[index]["source_split"] == "train"],
            dtype=int,
        )
        evaluation = np.asarray(
            [index for index in target if rows[index]["source_split"] == "validation"],
            dtype=int,
        )
        return train, evaluation
    labels = np.asarray([rows[index]["label"] == "active_tb" for index in target])
    return train_test_split(target, test_size=0.50, random_state=seed, stratify=labels)


def _model(head_name: str, seed: int) -> Pipeline:
    if head_name == "logistic":
        classifier: LogisticRegression | MLPClassifier = LogisticRegression(
            max_iter=2000, random_state=seed
        )
    else:
        classifier = MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=1e-3,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        )
    return Pipeline((("standardize", StandardScaler()), ("classifier", classifier)))


def _mean_sd_range(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "sample_sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _multi_seed_b0(
    features: np.ndarray,
    rows: list[dict[str, str]],
    target_site: str,
    aliases: set[tuple[str, str]],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    source_sites = (
        PRIMARY_SOURCE_SITES if target_site == "TBX11K" else SECONDARY_SOURCE_SITES
    )
    per_seed: dict[str, list[dict[str, Any]]] = {"logistic": [], "mlp": []}
    for seed in seeds:
        train, validation, _ = _source_split(rows, source_sites, seed)
        train = np.asarray(
            [
                index
                for index in train
                if (rows[index]["site"], rows[index]["image_id"]) not in aliases
            ],
            dtype=int,
        )
        validation = np.asarray(
            [
                index
                for index in validation
                if (rows[index]["site"], rows[index]["image_id"]) not in aliases
            ],
            dtype=int,
        )
        target_train, target_evaluation = _target_split(rows, target_site, seed)
        target_evaluation = np.asarray(
            [
                index
                for index in target_evaluation
                if (rows[index]["site"], rows[index]["image_id"]) not in aliases
            ],
            dtype=int,
        )
        labels = np.asarray(
            [rows[index]["label"] == "active_tb" for index in target_evaluation],
            dtype=int,
        )
        for head_name in ("logistic", "mlp"):
            head, source_scores, source_labels, threshold = _fit_head(
                head_name,
                features,
                rows,
                train,
                validation,
                target_site,
                source_sites,
                seed,
            )
            target_scores = _scores(head, head_name, features[target_evaluation])
            metrics = evaluate_threshold(target_scores, labels, threshold)
            source_metrics = evaluate_threshold(source_scores, source_labels, threshold)
            per_seed[head_name].append(
                {
                    "seed": seed,
                    "source_train_images": int(train.size),
                    "source_validation_images": int(validation.size),
                    "target_calibration_split_images": int(target_train.size),
                    "target_evaluation_images_after_duplicate_alias_exclusion": int(
                        target_evaluation.size
                    ),
                    "source_threshold": float(threshold),
                    "source_validation_metrics": source_metrics.__dict__,
                    "target_b0_metrics": metrics.__dict__,
                }
            )
    summaries: dict[str, Any] = {}
    for head_name, values in per_seed.items():
        summaries[head_name] = {
            "per_seed": values,
            "target_sensitivity": _mean_sd_range(
                [float(row["target_b0_metrics"]["sensitivity"]) for row in values]
            ),
            "target_specificity": _mean_sd_range(
                [float(row["target_b0_metrics"]["specificity"]) for row in values]
            ),
            "source_threshold": _mean_sd_range(
                [float(row["source_threshold"]) for row in values]
            ),
        }
    return {
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "source_sites": list(source_sites),
        "exact_pixel_duplicate_aliases_excluded_from_target_evaluation": len(aliases),
        "heads": summaries,
    }


def _nested_summaries(
    scores_positive: np.ndarray,
    scores_negative: np.ndarray,
    repetitions: int,
    seed: int,
    ltt_repetitions: int,
    source_validation_scores: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    tier_a = TierACalibrator(TARGET_FNR, FAILURE_PROBABILITY)
    crc = MarginalCRCCalibrator(TARGET_FNR)
    ltt_thresholds = np.unique(
        np.quantile(source_validation_scores, np.linspace(0.1, 0.9, 5))
    )
    if not ltt_thresholds.size:
        raise ValueError("Source-validation scores cannot define an IW-LTT grid.")

    counts: dict[int, dict[str, dict[str, Any]]] = {}
    for size in CALIBRATION_SIZES:
        counts[size] = {
            method: {
                "issued": 0,
                "refused": 0,
                "violations": 0,
                "sensitivities": [],
                "specificities": [],
            }
            for method in ("tier_a", "crc", "iw_ltt")
        }

    positive_count = scores_positive.size
    negative_count = scores_negative.size
    for repetition in range(repetitions):
        positive_order = rng.permutation(positive_count)
        negative_order = rng.permutation(negative_count)
        positive_calibration = scores_positive[positive_order[: positive_count // 2]]
        positive_evaluation = scores_positive[positive_order[positive_count // 2 :]]
        negative_evaluation = scores_negative[negative_order[negative_count // 2 :]]
        for size in CALIBRATION_SIZES:
            if size > positive_calibration.size:
                for method in counts[size].values():
                    method["refused"] += 1
                continue
            draw = rng.choice(positive_calibration, size=size, replace=False)
            tier_certificate = tier_a.calibrate(draw)
            crc_certificate = crc.calibrate(draw)
            for method_name, certificate, accepted in (
                (
                    "tier_a",
                    tier_certificate,
                    tier_certificate.status is CertificateStatus.CERTIFIED,
                ),
                (
                    "crc",
                    crc_certificate,
                    crc_certificate.status is RiskControlStatus.ACCEPTED,
                ),
            ):
                summary = counts[size][method_name]
                if not accepted:
                    summary["refused"] += 1
                    continue
                threshold = float(certificate.threshold)
                sensitivity = float(np.mean(positive_evaluation >= threshold))
                specificity = float(np.mean(negative_evaluation < threshold))
                summary["issued"] += 1
                summary["violations"] += int(1 - sensitivity > TARGET_FNR)
                summary["sensitivities"].append(sensitivity)
                summary["specificities"].append(specificity)

            if repetition < ltt_repetitions:
                ltt_certificate = ImportanceWeightedLTTCalibrator(
                    target_fnr=TARGET_FNR,
                    failure_probability=FAILURE_PROBABILITY,
                    permutations=10,
                    seed=seed + repetition,
                ).calibrate(
                    draw,
                    np.ones(size),
                    weight_upper_bound=1.0,
                    thresholds=ltt_thresholds,
                )
                summary = counts[size]["iw_ltt"]
                if ltt_certificate.status is RiskControlStatus.REFUSE:
                    summary["refused"] += 1
                else:
                    assert ltt_certificate.threshold is not None
                    threshold = ltt_certificate.threshold
                    sensitivity = float(np.mean(positive_evaluation >= threshold))
                    specificity = float(np.mean(negative_evaluation < threshold))
                    summary["issued"] += 1
                    summary["violations"] += int(1 - sensitivity > TARGET_FNR)
                    summary["sensitivities"].append(sensitivity)
                    summary["specificities"].append(specificity)
            elif repetition == ltt_repetitions:
                counts[size]["iw_ltt"]["not_run_repetitions"] = (
                    repetitions - ltt_repetitions
                )

    result: dict[str, Any] = {}
    for size, methods in counts.items():
        result[str(size)] = {}
        for method_name, summary in methods.items():
            issued = summary["issued"]
            total = issued + summary["refused"]
            violations = summary["violations"]
            result[str(size)][method_name] = {
                "repetitions": total,
                "issued": issued,
                "refused": summary["refused"],
                "refusal_rate": summary["refused"] / total if total else None,
                "violations": violations,
                "violation_rate": violations / issued if issued else None,
                "violation_rate_cp95": (
                    clopper_pearson_interval(violations, issued).__dict__
                    if issued
                    else None
                ),
                "mean_evaluation_sensitivity": (
                    float(np.mean(summary["sensitivities"]))
                    if summary["sensitivities"]
                    else None
                ),
                "mean_evaluation_specificity": (
                    float(np.mean(summary["specificities"]))
                    if summary["specificities"]
                    else None
                ),
                "theoretical_tier_a_failure_probability": (
                    float(binom.cdf(tier_a.allowed_misses(size), size, TARGET_FNR))
                    if tier_a.allowed_misses(size) is not None
                    else None
                ),
                "calibration_positive_count": size,
                "calibration_pool_size_per_repetition": positive_count // 2,
                "evaluation_positive_count_per_repetition": (
                    positive_count - positive_count // 2
                ),
            }
    return {
        "design": (
            "Each repetition randomly partitions all target positive images into "
            "disjoint calibration and evaluation halves, then samples n+ positives "
            "without replacement from the calibration half. Negative evaluation "
            "images are independently partitioned for each repetition."
        ),
        "repetitions": repetitions,
        "iw_ltt_repetitions": ltt_repetitions,
        "iw_ltt_random_orderings": 10,
        "iw_ltt_fixed_thresholds": [float(x) for x in ltt_thresholds],
        "calibration_sizes": list(CALIBRATION_SIZES),
        "unit_weight_w_crc_note": (
            "W-CRC with unit calibration/test weights equals CRC and is omitted as "
            "a duplicate method row."
        ),
        "results": result,
    }


def _domain_ratio_diagnostic(
    source_positive_features: np.ndarray,
    target_positive_features: np.ndarray,
    seed: int,
    folds: int = 5,
) -> dict[str, Any]:
    features = np.concatenate((source_positive_features, target_positive_features))
    domain = np.concatenate(
        (
            np.zeros(source_positive_features.shape[0], dtype=int),
            np.ones(target_positive_features.shape[0], dtype=int),
        )
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.empty(domain.size, dtype=float)
    for train, test in splitter.split(features, domain):
        base = Pipeline(
            (
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                ),
            )
        )
        calibrated = CalibratedClassifierCV(
            estimator=base, method="sigmoid", cv=3, ensemble=True
        )
        calibrated.fit(features[train], domain[train])
        probabilities[test] = calibrated.predict_proba(features[test])[:, 1]

    epsilon = 1e-8
    p_target = np.clip(probabilities, epsilon, 1 - epsilon)
    ratios = (p_target / (1 - p_target)) * (
        source_positive_features.shape[0] / target_positive_features.shape[0]
    )
    target_ratios = ratios[domain == 1]
    effective_sample_size = float(np.sum(target_ratios) ** 2 / np.sum(target_ratios**2))
    predicted_domain = (p_target >= 0.5).astype(int)
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss

    return {
        "purpose": "cross-fitted diagnostic of positive-class site separability",
        "domain_classifier": (
            "standardized logistic regression with sigmoid calibration and "
            "5-fold outer cross-fitting"
        ),
        "cross_validated_balanced_accuracy": float(
            balanced_accuracy_score(domain, predicted_domain)
        ),
        "cross_validated_auc": float(
            __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                domain, p_target
            )
        ),
        "cross_validated_brier_score": float(brier_score_loss(domain, p_target)),
        "cross_validated_log_loss": float(log_loss(domain, p_target)),
        "target_to_source_positive_density_ratio_diagnostic": {
            "quantiles": {
                str(q): float(np.quantile(target_ratios, q))
                for q in (0, 0.5, 0.9, 0.95, 0.99, 1)
            },
            "effective_sample_size": effective_sample_size,
            "target_sample_count": int(target_ratios.size),
            "effective_sample_size_fraction": effective_sample_size
            / target_ratios.size,
        },
        "valid_shift_weight_claim": False,
        "interpretation_limit": (
            "Cross-fitted domain odds are a diagnostic estimate, not a verified "
            "class-conditional density ratio or a known upper weight bound. They "
            "must not be supplied to a guaranteed weighted procedure."
        ),
    }


def _within_site_control(
    project_root: Path,
    features: np.ndarray,
    rows: list[dict[str, str]],
    aliases: set[tuple[str, str]],
    seed: int,
) -> dict[str, Any]:
    records: list[dict[str, str]] = []
    summaries: dict[str, Any] = {}
    for site in sorted({row["site"] for row in rows}):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["site"] == site and (row["site"], row["image_id"]) not in aliases
            ],
            dtype=int,
        )
        labels = np.asarray([rows[index]["label"] == "active_tb" for index in indices])
        development, test = train_test_split(
            indices, test_size=0.20, random_state=seed, stratify=labels
        )
        development_labels = np.asarray(
            [rows[index]["label"] == "active_tb" for index in development]
        )
        train, validation = train_test_split(
            development,
            test_size=0.25,
            random_state=seed,
            stratify=development_labels,
        )
        train_labels = np.asarray(
            [rows[index]["label"] == "active_tb" for index in train]
        )
        validation_labels = np.asarray(
            [rows[index]["label"] == "active_tb" for index in validation]
        )
        test_labels = np.asarray(
            [rows[index]["label"] == "active_tb" for index in test]
        )
        site_summary: dict[str, Any] = {
            "record_count_after_exact_pixel_alias_exclusion": int(indices.size),
            "heads": {},
        }
        train_set = set(train.tolist())
        validation_set = set(validation.tolist())
        from tb_certify.calibration.source_threshold import (
            EmpiricalSensitivityThreshold,
        )

        for head_name in ("logistic", "mlp"):
            head = _model(head_name, seed)
            head.fit(features[train], train_labels)
            validation_scores = head.predict_proba(features[validation])[:, 1]
            threshold = (
                EmpiricalSensitivityThreshold(0.90)
                .fit(validation_scores[validation_labels])
                .threshold
            )
            test_scores = head.predict_proba(features[test])[:, 1]
            site_summary["heads"][head_name] = evaluate_threshold(
                test_scores, test_labels.astype(int), threshold
            ).__dict__
        summaries[site] = site_summary
        for index in indices:
            role = (
                "model_train"
                if index in train_set
                else "source_validation"
                if index in validation_set
                else "evaluation"
            )
            records.append(
                {
                    "site": site,
                    "image_id": rows[index]["image_id"],
                    "label": rows[index]["label"],
                    "role": role,
                    "split_unit": (
                        "within-site image-level random split; patient linkage "
                        "not verified"
                    ),
                    "seed": str(seed),
                }
            )
    manifest = (
        project_root / "data" / "splits" / f"within_site_random_control_seed{seed}.csv"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return {
        "purpose": "within-site random split negative control",
        "split_manifest": str(manifest),
        "split_unit": "image-level; patient linkage not verified",
        "heads_by_site": summaries,
        "interpretation_limit": (
            "This split tests within-site interpolation and is not a transportability "
            "or patient-level validation study."
        ),
    }


def _wsr_type_i_simulation(repetitions: int, seed: int, n: int = 200) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    false_rejections = 0
    calibrator = ImportanceWeightedLTTCalibrator(
        target_fnr=TARGET_FNR,
        failure_probability=FAILURE_PROBABILITY,
        permutations=50,
        seed=seed,
    )
    for _ in range(repetitions):
        losses = rng.binomial(1, TARGET_FNR, size=n).astype(float)
        orderings = tuple(rng.permutation(n) for _ in range(50))
        p_value = calibrator._wsr_p_value(
            losses,
            TARGET_FNR,
            FAILURE_PROBABILITY,
            orderings,
        )
        false_rejections += int(p_value <= FAILURE_PROBABILITY)
    interval = clopper_pearson_interval(false_rejections, repetitions)
    return {
        "null_risk": TARGET_FNR,
        "failure_probability": FAILURE_PROBABILITY,
        "sample_size": n,
        "repetitions": repetitions,
        "random_orderings_per_replication": 50,
        "false_rejections": false_rejections,
        "empirical_type_i_rate": false_rejections / repetitions,
        "type_i_rate_cp95": interval.__dict__,
        "interpretation_limit": (
            "Monte Carlo calibration check under iid Bernoulli null losses; this "
            "does not validate the covariate-shift ratio assumptions."
        ),
    }


def run_audits(
    project_root: Path,
    embedding_dir: Path,
    output_dir: Path,
    repetitions: int,
    ltt_repetitions: int,
    type_i_repetitions: int,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    project_root = project_root.resolve()
    embedding_dir = embedding_dir.resolve()
    rows = _read_csv(embedding_dir / "manifest.csv")
    features = np.load(embedding_dir / "embeddings.npy", mmap_mode="r")
    metadata = json.loads((embedding_dir / "metadata.json").read_text(encoding="utf-8"))
    if features.ndim != 2 or features.shape[0] != len(rows):
        raise ValueError("Embedding rows must align one-to-one with the manifest.")
    aliases = _exact_duplicate_aliases(project_root)

    results: dict[str, Any] = {
        "study": "multi-seed and nested target-resampling robustness audits",
        "encoder": metadata,
        "seed": 42,
        "split_unit": "image-level; patient linkage not verified",
        "patient_level_claim": patient_level_supported(project_root),
        "pac_claim": False,
        "exact_pixel_duplicate_policy": (
            "Keep the first deterministic identity-audit record in each exact-pixel "
            "duplicate group; exclude the remaining aliases from target evaluations "
            "and within-site controls."
        ),
        "exact_pixel_duplicate_alias_count": len(aliases),
        "multi_seed_b0": {},
        "nested_target_resampling": {},
        "density_ratio_diagnostic": {},
        "within_site_control": _within_site_control(
            project_root, features, rows, aliases, seed=42
        ),
        "wsr_type_i_simulation": _wsr_type_i_simulation(type_i_repetitions, seed=4200),
    }

    for target_site in ("TBX11K", "Shenzhen"):
        results["multi_seed_b0"][target_site] = _multi_seed_b0(
            features, rows, target_site, aliases, seeds
        )
        source_sites = (
            PRIMARY_SOURCE_SITES if target_site == "TBX11K" else SECONDARY_SOURCE_SITES
        )
        source_train, source_validation, _ = _source_split(rows, source_sites, 42)
        source_train = np.asarray(
            [
                index
                for index in source_train
                if (rows[index]["site"], rows[index]["image_id"]) not in aliases
            ],
            dtype=int,
        )
        source_validation = np.asarray(
            [
                index
                for index in source_validation
                if (rows[index]["site"], rows[index]["image_id"]) not in aliases
            ],
            dtype=int,
        )
        positive_source = np.asarray(
            [rows[index]["label"] == "active_tb" for index in source_train], dtype=bool
        )
        positive_target_indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["site"] == target_site and row["label"] == "active_tb"
            ],
            dtype=int,
        )
        negative_target_indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["site"] == target_site
                and row["label"] != "active_tb"
                and (row["site"], row["image_id"]) not in aliases
            ],
            dtype=int,
        )
        results["nested_target_resampling"][target_site] = {}
        for head_name in ("logistic", "mlp"):
            head, source_validation_scores, _, _ = _fit_head(
                head_name,
                features,
                rows,
                source_train,
                source_validation,
                target_site,
                source_sites,
                42,
            )
            positive_scores = _scores(
                head, head_name, features[positive_target_indices]
            )
            negative_scores = _scores(
                head, head_name, features[negative_target_indices]
            )
            nested = _nested_summaries(
                positive_scores,
                negative_scores,
                repetitions,
                seed=4200 + (0 if target_site == "TBX11K" else 100),
                ltt_repetitions=ltt_repetitions,
                source_validation_scores=source_validation_scores,
            )
            results["nested_target_resampling"][target_site][head_name] = {
                "target_positive_images": int(positive_scores.size),
                "target_negative_images_after_exact_pixel_alias_exclusion": int(
                    negative_scores.size
                ),
                "resampling": nested,
            }
        source_positive_features = features[source_train[positive_source]]
        results["density_ratio_diagnostic"][target_site] = _domain_ratio_diagnostic(
            source_positive_features,
            features[positive_target_indices],
            seed=42,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    results["output_path"] = str(output_path)
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--ltt-repetitions", type=int, default=20)
    parser.add_argument("--type-i-repetitions", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    arguments = parser.parse_args()
    if arguments.repetitions < 1:
        parser.error("--repetitions must be positive")
    if not 1 <= arguments.ltt_repetitions <= arguments.repetitions:
        parser.error("--ltt-repetitions must lie between one and repetitions")
    if arguments.type_i_repetitions < 1:
        parser.error("--type-i-repetitions must be positive")
    if len(set(arguments.seeds)) != len(arguments.seeds):
        parser.error("--seeds must be unique")
    result = run_audits(
        arguments.project_root,
        arguments.embedding_dir,
        arguments.output_dir,
        arguments.repetitions,
        arguments.ltt_repetitions,
        arguments.type_i_repetitions,
        tuple(arguments.seeds),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
