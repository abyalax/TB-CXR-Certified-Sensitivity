"""Deterministic grayscale CXR conversion for frozen image encoders."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class CXRPreprocessingConfig:
    image_size: int = 224
    low_percentile_16bit: float = 0.5
    high_percentile_16bit: float = 99.5
    resize_policy: str = "square_lanczos"
    output_channels: int = 3

    def __post_init__(self) -> None:
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16 pixels.")
        if not 0 <= self.low_percentile_16bit < self.high_percentile_16bit <= 100:
            raise ValueError(
                "16-bit window percentiles must satisfy 0 <= low < high <= 100."
            )
        if self.resize_policy != "square_lanczos":
            raise ValueError(
                "Only the prespecified square_lanczos resize is supported."
            )
        if self.output_channels != 3:
            raise ValueError(
                "The current encoder interface requires 3 output channels."
            )


class CXRImagePreprocessor:
    """Convert a CXR to uint8 RGB at a fixed square resolution.

    Existing 8-bit images retain their intensity mapping. Higher-bit grayscale
    images use the configured per-image percentile window; this choice is part
    of the experiment configuration and must be held fixed across sites.
    """

    HIGH_BIT_MODES = {"I", "F", "I;16", "I;16L", "I;16B", "I;16N"}

    def __init__(self, config: CXRPreprocessingConfig | None = None) -> None:
        self.config = config or CXRPreprocessingConfig()

    def process_path(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        with Image.open(path) as image:
            image.load()
            return self.process_image(image)

    def process_bytes(self, image_bytes: bytes) -> np.ndarray:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            return self.process_image(image)

    def process_image(self, image: Image.Image) -> np.ndarray:
        image = ImageOps.exif_transpose(image)
        if image.mode in self.HIGH_BIT_MODES:
            gray = self._window_high_bit_grayscale(image)
        else:
            gray = image.convert("L")
        resized = gray.resize(
            (self.config.image_size, self.config.image_size),
            resample=Image.Resampling.LANCZOS,
        )
        gray_array = np.asarray(resized, dtype=np.uint8)
        return np.repeat(gray_array[..., np.newaxis], 3, axis=2)

    def _window_high_bit_grayscale(self, image: Image.Image) -> Image.Image:
        values = np.asarray(image)
        if values.ndim != 2:
            raise ValueError(
                "Higher-bit input must be a single-channel grayscale image."
            )
        values = values.astype(np.float64, copy=False)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError("Image has no finite intensity values.")
        low, high = np.percentile(
            finite_values,
            [
                self.config.low_percentile_16bit,
                self.config.high_percentile_16bit,
            ],
        )
        if high <= low:
            scaled = np.zeros(values.shape, dtype=np.uint8)
        else:
            finite_values = np.where(np.isfinite(values), values, low)
            normalized = np.clip((finite_values - low) / (high - low), 0, 1)
            scaled = np.rint(normalized * 255).astype(np.uint8)
        return Image.fromarray(scaled)
