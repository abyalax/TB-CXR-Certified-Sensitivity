"""Frozen feature extraction for revision-pinned image encoders."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tb_certify.models.provenance import EncoderProvenance


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_manifest(manifest_path: Path, cache_root: Path) -> list[dict[str, str]]:
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Image manifest is empty: {manifest_path}")
    root = cache_root.resolve()
    for row in rows:
        cache_path = (root / row["cache_path"]).resolve()
        if root not in cache_path.parents:
            raise ValueError(f"Cache path escapes the image root: {row['cache_path']}")
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
    return rows


def _load_rgb(path: Path, expected_size: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.size != (expected_size, expected_size):
        raise ValueError(
            f"Expected {expected_size}x{expected_size} cache image, got "
            f"{image.width}x{image.height}: {path}"
        )
    return image


def _choose_device(torch: Any, device_name: str) -> Any:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested, but PyTorch cannot access a CUDA GPU."
            )
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_encoder(
    encoder_key: str, spec: dict[str, Any], device: Any
) -> tuple[Any, Any, Any]:
    if encoder_key == "general":
        try:
            import timm
            from timm.data import resolve_model_data_config
        except ImportError as error:
            raise RuntimeError(
                "The general encoder requires timm; install `uv sync --extra vision`."
            ) from error

        hub_model = f"hf-hub:{spec['model_name']}@{spec['revision']}"
        model = timm.create_model(hub_model, pretrained=True, num_classes=0)
        data_config = resolve_model_data_config(model)
        normalization = {
            "mean": data_config["mean"],
            "std": data_config["std"],
        }
        model = model.to(device).eval()
        return model, None, normalization

    if encoder_key == "chest_xray":
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise RuntimeError(
                "RAD-DINO requires transformers; install `uv sync --extra vision`."
            ) from error

        checkpoint_dir = spec.get("resolved_checkpoint_dir")
        source = checkpoint_dir or spec["model_name"]
        source_kwargs = (
            {"local_files_only": True}
            if checkpoint_dir
            else {"revision": spec["revision"]}
        )
        processor = AutoImageProcessor.from_pretrained(
            source, use_fast=False, **source_kwargs
        )
        model = AutoModel.from_pretrained(source, **source_kwargs)
        model = model.to(device).eval()
        return model, processor, None

    raise ValueError(f"Unsupported encoder key: {encoder_key}")


def _prepare_general_batch(
    images: list[Image.Image], normalization: dict[str, list[float]], device: Any
) -> Any:
    import torch

    arrays = np.stack([np.asarray(image, dtype=np.uint8) for image in images])
    batch = torch.from_numpy(arrays.copy()).permute(0, 3, 1, 2).to(device=device)
    batch = batch.to(dtype=torch.float32).div_(255.0)
    mean = torch.tensor(normalization["mean"], dtype=batch.dtype, device=device).view(
        1, 3, 1, 1
    )
    std = torch.tensor(normalization["std"], dtype=batch.dtype, device=device).view(
        1, 3, 1, 1
    )
    return (batch - mean) / std


def _extract_batch(
    encoder_key: str,
    model: Any,
    processor: Any,
    normalization: Any,
    images: list[Image.Image],
    device: Any,
) -> Any:
    import torch

    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if encoder_key == "chest_xray" and device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), amp_context:
        if encoder_key == "general":
            output = model(_prepare_general_batch(images, normalization, device))
            features = output
        else:
            inputs = processor(images=images, return_tensors="pt")
            pixels = inputs["pixel_values"].to(device=device)
            output = model(pixel_values=pixels)
            features = output.last_hidden_state[:, 0, :]
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError(
            "The encoder must produce one two-dimensional feature per image."
        )
    return features.detach().to(device="cpu", dtype=torch.float32).numpy()


def extract_frozen_embeddings(
    *,
    encoder_key: str,
    spec: dict[str, Any],
    manifest_path: Path,
    cache_root: Path,
    output_dir: Path,
    device_name: str = "auto",
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Extract aligned image embeddings and write their provenance sidecars."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; choose a new directory: {output_dir}"
        )
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    provenance = EncoderProvenance(
        model_name=spec["model_name"],
        revision=spec["revision"],
        license_name=spec.get("license_name"),
        license_verified=bool(spec.get("license_verified")),
        pretraining_sites=tuple(spec["pretraining_sites"])
        if spec.get("pretraining_sites") is not None
        else None,
        pretraining_data_reviewed=bool(spec.get("pretraining_data_reviewed")),
    )
    for target in ("Montgomery", "Shenzhen", "TBX11K"):
        provenance.validate_for_target(target)

    import torch

    device = _choose_device(torch, device_name)
    if batch_size is None:
        if encoder_key == "chest_xray":
            batch_size = 4 if device.type == "cuda" else 1
        else:
            batch_size = 64 if device.type == "cuda" else 16

    image_size = int(spec["input_resolution"])
    rows = _read_manifest(manifest_path, cache_root)
    cache_metadata_path = cache_root / "preprocessing.json"
    with cache_metadata_path.open(encoding="utf-8") as stream:
        cache_metadata = json.load(stream)
    if cache_metadata["preprocessing"]["image_size"] != image_size:
        raise ValueError(
            f"Encoder expects {image_size}px inputs, but cache metadata records "
            f"{cache_metadata['preprocessing']['image_size']}px."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    if embeddings_path.exists():
        raise FileExistsError(embeddings_path)

    model, processor, normalization = _load_encoder(encoder_key, spec, device)
    feature_dim = (
        int(model.num_features)
        if encoder_key == "general"
        else int(model.config.hidden_size)
    )
    temporary_embeddings = output_dir / "embeddings.npy.tmp"
    features = np.lib.format.open_memmap(
        temporary_embeddings,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), feature_dim),
    )
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            images = [
                _load_rgb(cache_root / row["cache_path"], image_size)
                for row in batch_rows
            ]
            batch_features = _extract_batch(
                encoder_key,
                model,
                processor,
                normalization,
                images,
                device,
            )
            if batch_features.shape != (len(batch_rows), feature_dim):
                raise ValueError(
                    "Encoder output shape differs from the configured feature width: "
                    f"{batch_features.shape} vs {(len(batch_rows), feature_dim)}"
                )
            features[start : start + len(batch_rows)] = batch_features
            if (start // batch_size + 1) % 100 == 0:
                features.flush()
                print(f"Extracted {start + len(batch_rows)}/{len(rows)} images.")
        features.flush()
    except Exception:
        del features
        temporary_embeddings.unlink(missing_ok=True)
        raise
    del features
    os.replace(temporary_embeddings, embeddings_path)
    rows_path = output_dir / "manifest.csv"
    shutil.copyfile(manifest_path, rows_path)

    metadata = {
        "encoder_key": encoder_key,
        "model_name": spec["model_name"],
        "revision": spec["revision"],
        "checkpoint_source": spec.get("resolved_checkpoint_dir", spec["model_name"]),
        "checkpoint_sha256": _sha256(
            Path(spec["resolved_checkpoint_dir"]) / "model.safetensors"
        )
        if spec.get("resolved_checkpoint_dir")
        else None,
        "license_name": spec.get("license_name"),
        "pretraining_sites": spec["pretraining_sites"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "encoder_library": importlib.metadata.version(
                "timm" if encoder_key == "general" else "transformers"
            ),
        },
        "batch_size": batch_size,
        "inference_precision": (
            "float16_autocast"
            if encoder_key == "chest_xray" and device.type == "cuda"
            else "float32"
        ),
        "input_resolution": image_size,
        "image_count": len(rows),
        "feature_dimension": feature_dim,
        "feature_pooling": "global_average_pooling"
        if encoder_key == "general"
        else "cls_token",
        "input_normalization": normalization
        if encoder_key == "general"
        else {
            "mean": processor.image_mean,
            "std": processor.image_std,
            "size": processor.size,
            "crop_size": processor.crop_size,
            "resample": str(processor.resample),
            "do_resize": processor.do_resize,
            "do_center_crop": processor.do_center_crop,
            "do_rescale": processor.do_rescale,
            "rescale_factor": processor.rescale_factor,
            "do_normalize": processor.do_normalize,
            "do_convert_rgb": processor.do_convert_rgb,
            "use_fast": bool(getattr(processor, "is_fast", False)),
        },
        "cache_root": str(cache_root.resolve()),
        "cache_metadata_sha256": _sha256(cache_metadata_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "embedding_sha256": _sha256(embeddings_path),
        "embedding_path": str(embeddings_path.resolve()),
        "manifest_path": str(rows_path.resolve()),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def extract_from_project(
    *,
    project_root: Path,
    encoder_key: str,
    batch_size: int | None,
    device_name: str,
    output_dir: Path | None,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Resolve project configuration and launch the selected encoder run."""

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "YAML configuration needs PyYAML; install `uv sync --extra vision`."
        ) from error

    root = project_root.resolve()
    config_path = root / "configs" / "model.yaml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    try:
        spec = config["encoders"][encoder_key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"No encoder configuration for {encoder_key!r}.") from error
    spec = dict(spec)
    local_checkpoint_dir = spec.get("local_checkpoint_dir")
    if local_checkpoint_dir:
        checkpoint_path = (root / local_checkpoint_dir).resolve()
        if checkpoint_path.is_dir():
            spec["resolved_checkpoint_dir"] = str(checkpoint_path)
            checkpoint = spec.get("checkpoint", {})
            weight_path = checkpoint_path / checkpoint.get(
                "weight_file", "model.safetensors"
            )
            if not weight_path.is_file():
                raise FileNotFoundError(weight_path)
            expected_size = checkpoint.get("size_bytes")
            if (
                expected_size is not None
                and weight_path.stat().st_size != expected_size
            ):
                raise ValueError(
                    f"Checkpoint size mismatch: expected {expected_size}, "
                    f"got {weight_path.stat().st_size}."
                )
            expected_hash = checkpoint.get("sha256")
            if expected_hash is not None:
                actual_hash = _sha256(weight_path)
                if actual_hash.casefold() != expected_hash.casefold():
                    raise ValueError(
                        "Checkpoint SHA-256 does not match the pinned config."
                    )

    if cache_root is None:
        cache_name = "rad_dino_518" if encoder_key == "chest_xray" else "primary"
        cache_root = root / "data" / "processed" / cache_name
    elif not cache_root.is_absolute():
        cache_root = root / cache_root
    cache_root = cache_root.resolve()
    manifest_path = cache_root / "manifest.csv"
    destination = output_dir or (
        root
        / "artifacts"
        / "embeddings"
        / f"{encoder_key}-{str(spec['revision'])[:12]}"
    )
    if not destination.is_absolute():
        destination = root / destination
    return extract_frozen_embeddings(
        encoder_key=encoder_key,
        spec=spec,
        manifest_path=manifest_path,
        cache_root=cache_root,
        output_dir=destination,
        device_name=device_name,
        batch_size=batch_size,
    )
