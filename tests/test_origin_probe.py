"""Tests for the site-origin shortcut probe."""

from __future__ import annotations

import csv

import numpy as np
from scripts.probe_dataset_origin import run_probe


def test_origin_probe_reports_cross_validated_metrics(tmp_path) -> None:
    embedding_dir = tmp_path / "embeddings"
    embedding_dir.mkdir()
    features = np.vstack(
        (
            np.tile([0.0, 0.0], (5, 1)),
            np.tile([5.0, 5.0], (5, 1)),
        )
    )
    np.save(embedding_dir / "embeddings.npy", features)
    with (embedding_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["site"])
        writer.writeheader()
        writer.writerows([{"site": "A"}] * 5 + [{"site": "B"}] * 5)

    result = run_probe(embedding_dir, seed=3, folds=5)

    assert result["cross_validated_balanced_accuracy"] == 1.0
    assert result["cross_validated_macro_ovr_auc"] == 1.0
