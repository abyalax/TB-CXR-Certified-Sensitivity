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


def test_origin_probe_excludes_exact_pixel_duplicate_aliases(tmp_path) -> None:
    embedding_dir = tmp_path / "embeddings"
    embedding_dir.mkdir()
    features = np.asarray(
        [[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.0, 5.0], [5.1, 5.1]],
        dtype=float,
    )
    np.save(embedding_dir / "embeddings.npy", features)
    with (embedding_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["site", "image_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"site": "A", "image_id": "a1"},
                {"site": "A", "image_id": "a2"},
                {"site": "B", "image_id": "b1"},
                {"site": "B", "image_id": "b2"},
                {"site": "B", "image_id": "b3"},
            ]
        )
    audit_path = tmp_path / "identity.json"
    audit_path.write_text(
        '{"exact_decoded_rgb_pixel_duplicates":{"B":{"duplicate_groups":[["b1","b2"]]}}}',
        encoding="utf-8",
    )

    result = run_probe(
        embedding_dir,
        seed=3,
        folds=2,
        identity_audit_path=audit_path,
    )

    assert result["exact_pixel_duplicate_aliases_excluded"] == 1
    assert result["site_counts"] == {"A": 2, "B": 2}
