"""Train a source-only RAD-DINO LoRA head and audit it on TBX11K."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

from tb_certify.calibration.source_threshold import EmpiricalSensitivityThreshold
from tb_certify.evaluation.metrics import evaluate_threshold
from tb_certify.evaluation.tier_a_study import TierAStudy


class CXRDataset(Dataset[tuple[Image.Image, int]]):
    def __init__(self, root: Path, rows: list[dict[str, str]], indices: np.ndarray):
        self.root = root
        self.rows = rows
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Image.Image, int]:
        row = self.rows[int(self.indices[item])]
        with Image.open(self.root / row["cache_path"]) as image:
            return image.convert("RGB"), int(row["label"] == "active_tb")


class LoRAClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.encoder(pixel_values=pixel_values)
        return self.classifier(output.last_hidden_state[:, 0].float()).squeeze(-1)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _source_split(
    rows: list[dict[str, str]], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    train: list[int] = []
    validation: list[int] = []
    for site in ("Montgomery", "Shenzhen"):
        site_indices = np.asarray(
            [i for i, row in enumerate(rows) if row["site"] == site], dtype=int
        )
        labels = np.asarray([rows[i]["label"] == "active_tb" for i in site_indices])
        site_train, site_validation = train_test_split(
            site_indices, test_size=0.20, random_state=seed, stratify=labels
        )
        train.extend(site_train.tolist())
        validation.extend(site_validation.tolist())
    return np.asarray(train), np.asarray(validation)


def _write_source_manifest(
    path: Path,
    rows: list[dict[str, str]],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_set = set(train_indices.tolist())
    fields = ("site", "image_id", "role", "split_unit", "seed")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in sorted(train_set.union(validation_indices.tolist())):
            writer.writerow(
                {
                    "site": rows[index]["site"],
                    "image_id": rows[index]["image_id"],
                    "role": (
                        "source_train" if index in train_set else "source_validation"
                    ),
                    "split_unit": "patient-level; one image represents one patient",
                    "seed": str(seed),
                }
            )


def _loader(
    dataset: CXRDataset,
    processor: Any,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)

    def collate(batch: list[tuple[Image.Image, int]]) -> dict[str, torch.Tensor]:
        images, labels = zip(*batch, strict=True)
        pixel_values = processor(images=list(images), return_tensors="pt").pixel_values
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.float32),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        generator=generator,
        collate_fn=collate,
    )


def _predict(
    model: LoRAClassifier, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    score_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(pixels)
            score_parts.append(torch.sigmoid(logits.float()).cpu().numpy())
            label_parts.append(batch["labels"].numpy().astype(int))
    return np.concatenate(score_parts), np.concatenate(label_parts)


def run(
    project_root: Path,
    embedding_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    gradient_accumulation: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("This fp16 LoRA ablation requires CUDA.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")

    metadata = json.loads((embedding_dir / "metadata.json").read_text(encoding="utf-8"))
    rows = _read_csv(embedding_dir / "manifest.csv")
    cache_root = Path(metadata["cache_root"])
    source_train, source_validation = _source_split(rows, seed)
    source_manifest_path = (
        project_root / "data" / "splits" / f"lora_source_seed{seed}.csv"
    )
    _write_source_manifest(
        source_manifest_path, rows, source_train, source_validation, seed
    )
    target_train = np.asarray(
        [
            i
            for i, row in enumerate(rows)
            if row["site"] == "TBX11K" and row["source_split"] == "train"
        ]
    )
    target_validation = np.asarray(
        [
            i
            for i, row in enumerate(rows)
            if row["site"] == "TBX11K" and row["source_split"] == "validation"
        ]
    )
    target_calibration_positive = np.asarray(
        [i for i in target_train if rows[i]["label"] == "active_tb"]
    )

    model_name = metadata["model_name"]
    revision = metadata["revision"]
    checkpoint = Path(metadata["checkpoint_source"])
    processor = AutoImageProcessor.from_pretrained(
        checkpoint if checkpoint.is_dir() else model_name,
        revision=revision,
        use_fast=False,
    )
    backbone = AutoModel.from_pretrained(
        checkpoint if checkpoint.is_dir() else model_name, revision=revision
    )
    hidden_size = int(backbone.config.hidden_size)
    backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["query", "value"],
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    model = LoRAClassifier(backbone, hidden_size).to(device)
    model.encoder.print_trainable_parameters()
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_loader = _loader(
        CXRDataset(cache_root, rows, source_train), processor, batch_size, True, seed
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01
    )
    criterion = nn.BCEWithLogitsLoss()
    epoch_losses: list[float] = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        batch_count = len(train_loader)
        for batch_index, batch in enumerate(train_loader):
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(pixels)
                loss = criterion(logits.float(), labels)
            (loss / gradient_accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            should_step = (batch_index + 1) % gradient_accumulation == 0 or (
                batch_index + 1 == batch_count
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        epoch_losses.append(float(np.mean(losses)))
        print(
            f"epoch={epoch + 1}/{epochs} mean_loss={epoch_losses[-1]:.6f}", flush=True
        )

    source_validation_scores, source_validation_labels = _predict(
        model,
        _loader(
            CXRDataset(cache_root, rows, source_validation),
            processor,
            batch_size,
            False,
            seed,
        ),
        device,
    )
    source_threshold = (
        EmpiricalSensitivityThreshold(0.90)
        .fit(source_validation_scores[source_validation_labels == 1])
        .threshold
    )
    evaluation_scores, evaluation_labels = _predict(
        model,
        _loader(
            CXRDataset(cache_root, rows, target_validation),
            processor,
            batch_size,
            False,
            seed,
        ),
        device,
    )
    calibration_scores, _ = _predict(
        model,
        _loader(
            CXRDataset(cache_root, rows, target_calibration_positive),
            processor,
            batch_size,
            False,
            seed,
        ),
        device,
    )
    tier_a = TierAStudy(target_specificity=0.70, confidence=0.95).run(
        calibration_positive_scores=calibration_scores,
        evaluation_scores=evaluation_scores,
        evaluation_labels=evaluation_labels,
        calibration_sizes=(29, 46, 61, 100, 200),
        repetitions=repetitions,
        seed=seed,
        split_unit=(
            "official TBX11K patient-level train/validation (one image per patient)"
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(output_dir / "adapter")
    classifier_head_path = output_dir / "classifier_head.pt"
    torch.save(model.classifier.state_dict(), classifier_head_path)
    results: dict[str, Any] = {
        "study": "source-only RAD-DINO LoRA ablation; exploratory patient-level (one image per patient)",
        "target_site": "TBX11K",
        "source_sites": ["Montgomery", "Shenzhen"],
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": batch_size * gradient_accumulation,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["query", "value"],
        },
        "trainable_parameters": trainable_parameters,
        "source_train_images": int(len(source_train)),
        "source_validation_images": int(len(source_validation)),
        "target_calibration_positive_images": int(len(target_calibration_positive)),
        "target_evaluation_images": int(len(target_validation)),
        "epoch_mean_bce_losses": epoch_losses,
        "source_validation_threshold": float(source_threshold),
        "source_validation_metrics": evaluate_threshold(
            source_validation_scores, source_validation_labels, source_threshold
        ).__dict__,
        "target_b0_metrics": evaluate_threshold(
            evaluation_scores, evaluation_labels, source_threshold
        ).__dict__,
        "tier_a_exploratory": tier_a.to_dict(),
        "encoder": metadata,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        },
        "adapter_path": str(output_dir / "adapter"),
        "classifier_head_path": str(classifier_head_path),
        "source_split_manifest": str(source_manifest_path),
        "patient_level_claim": True,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        min(args.epochs, args.batch_size, args.gradient_accumulation, args.repetitions)
        < 1
    ):
        parser.error(
            "epochs, batch-size, gradient-accumulation, and "
            "repetitions must be positive"
        )
    result = run(
        args.project_root.resolve(),
        args.embedding_dir.resolve(),
        args.output_dir.resolve(),
        args.epochs,
        args.batch_size,
        args.gradient_accumulation,
        args.repetitions,
        args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
