"""Focused checks for the preprocessing ablation options."""

from __future__ import annotations

import numpy as np
from PIL import Image

from tb_certify.data.preprocessing import CXRImagePreprocessor, CXRPreprocessingConfig


def test_letterbox_resize_preserves_aspect_ratio_with_padding() -> None:
    image = Image.new("L", (8, 4), color=120)
    processor = CXRImagePreprocessor(
        CXRPreprocessingConfig(image_size=32, resize_policy="letterbox_lanczos")
    )

    result = processor.process_image(image)

    assert result.shape == (32, 32, 3)
    assert np.all(result[:8] == 0)
    assert np.all(result[8:24] == 120)
    assert np.all(result[24:] == 0)


def test_window_all_grayscale_stretches_eight_bit_input() -> None:
    pixels = np.zeros((8, 8), dtype=np.uint8)
    pixels[:, 4:] = 200
    image = Image.fromarray(pixels)
    config = CXRPreprocessingConfig(
        image_size=32,
        low_percentile_16bit=0,
        high_percentile_16bit=100,
        window_all_grayscale=True,
    )

    result = CXRImagePreprocessor(config).process_image(image)

    assert result.min() == 0
    assert result.max() == 255


def test_uint8_histogram_window_matches_numpy_percentiles() -> None:
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 256, size=(257, 389), dtype=np.uint8)
    processor = CXRImagePreprocessor(CXRPreprocessingConfig())

    result = np.asarray(processor._window_grayscale(Image.fromarray(pixels)))
    low, high = np.percentile(pixels, (0.5, 99.5))
    expected = np.rint(
        np.clip((pixels.astype(np.float64) - low) / (high - low), 0, 1) * 255
    ).astype(np.uint8)

    np.testing.assert_array_equal(result, expected)
