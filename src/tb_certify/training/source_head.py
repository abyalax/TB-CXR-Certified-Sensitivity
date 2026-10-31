"""Small logistic head trained only on declared source-site embeddings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import isfinite

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tb_certify.calibration.source_threshold import (
    EmpiricalSensitivityThreshold,
    EmpiricalThreshold,
)


class SourceOnlyLogisticHead:
    """Fit a standardized logistic head without target-site labels or features.

    The caller must create source train/validation partitions at the verified
    sampling-unit level. Image-level splits do not establish patient separation.
    before fitting. Threshold selection is a separate source-validation step
    and is explicitly an empirical B0 baseline, not a certificate.
    """

    def __init__(
        self,
        regularization: float = 1.0,
        max_iterations: int = 2000,
        seed: int = 42,
    ) -> None:
        if not isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be positive.")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.seed = seed
        self._pipeline: Pipeline | None = None
        self.source_sites: frozenset[str] = frozenset()
        self.held_out_target_site: str | None = None

    def fit(
        self,
        features: np.ndarray,
        labels: Iterable[int],
        site_ids: Sequence[str],
        source_sites: Sequence[str],
        held_out_target_site: str,
    ) -> SourceOnlyLogisticHead:
        feature_array = np.asarray(features, dtype=np.float64)
        label_array = np.asarray(tuple(labels), dtype=int)
        site_array = np.asarray(tuple(site_ids), dtype=str)
        source_set = frozenset(source_sites)
        if feature_array.ndim != 2 or feature_array.shape[0] == 0:
            raise ValueError("features must be a non-empty two-dimensional matrix.")
        if not np.all(np.isfinite(feature_array)):
            raise ValueError("features must contain only finite values.")
        if label_array.shape != (feature_array.shape[0],):
            raise ValueError("labels length must match the feature rows.")
        if site_array.shape != (feature_array.shape[0],):
            raise ValueError("site_ids length must match the feature rows.")
        if not np.isin(label_array, [0, 1]).all():
            raise ValueError("labels must use 1 for active TB and 0 for non-TB.")
        if (
            not source_set
            or not held_out_target_site.strip()
            or any(not site.strip() for site in source_set)
            or held_out_target_site in source_set
        ):
            raise ValueError("Declare source sites and keep the target site held out.")
        observed_sites = set(site_array.tolist())
        unexpected_sites = observed_sites.difference(source_set)
        if unexpected_sites:
            raise ValueError(
                f"Training data contains non-source sites: {sorted(unexpected_sites)}"
            )
        missing_sources = source_set.difference(observed_sites)
        if missing_sources:
            raise ValueError(
                "Declared source sites have no training features: "
                f"{sorted(missing_sources)}"
            )
        if held_out_target_site in observed_sites:
            raise ValueError("Held-out target-site data must not enter head training.")
        if set(label_array.tolist()) != {0, 1}:
            raise ValueError("Source training data must contain both binary classes.")

        self._pipeline = Pipeline(
            steps=[
                ("standardize", StandardScaler()),
                (
                    "logistic_head",
                    LogisticRegression(
                        C=self.regularization,
                        max_iter=self.max_iterations,
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        self._pipeline.fit(feature_array, label_array)
        self.source_sites = source_set
        self.held_out_target_site = held_out_target_site
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("The logistic head must be fit before scoring.")
        feature_array = np.asarray(features, dtype=np.float64)
        if feature_array.ndim != 2 or not np.all(np.isfinite(feature_array)):
            raise ValueError("features must be a finite two-dimensional matrix.")
        return self._pipeline.decision_function(feature_array)

    def choose_source_threshold(
        self,
        validation_features: np.ndarray,
        validation_labels: Iterable[int],
        validation_site_ids: Sequence[str],
        target_sensitivity: float = 0.90,
    ) -> EmpiricalThreshold:
        if self._pipeline is None:
            raise RuntimeError("Fit the logistic head before choosing a threshold.")
        labels = np.asarray(tuple(validation_labels), dtype=int)
        sites = set(validation_site_ids)
        if labels.shape[0] != len(validation_site_ids):
            raise ValueError("Validation labels and site IDs must have equal length.")
        if not sites or not sites.issubset(self.source_sites):
            raise ValueError(
                "Threshold validation must use declared source sites only."
            )  # noqa: E501
        if labels.size == 0 or not np.isin(labels, [0, 1]).all():
            raise ValueError("Validation data must have binary labels.")
        validation_scores = self.score(validation_features)
        if validation_scores.shape[0] != labels.shape[0]:
            raise ValueError("Validation labels must match the feature rows.")
        positive_scores = validation_scores[labels == 1]
        return EmpiricalSensitivityThreshold(target_sensitivity).fit(positive_scores)
