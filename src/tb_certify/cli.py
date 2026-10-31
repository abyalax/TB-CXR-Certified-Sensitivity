"""Command-line entry points for inspecting study inputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from tb_certify.data.nlm import NLMChestXrayCatalog
from tb_certify.data.pipeline import PrimaryCohortPreprocessor
from tb_certify.data.preprocessing import CXRPreprocessingConfig
from tb_certify.data.roi import ROIAnnotationCsvReader
from tb_certify.data.tbx11k import TBX11KArchiveCatalog


def _inspect_data(project_root: Path) -> int:
    raw_dir = project_root / "data" / "raw"
    catalog = TBX11KArchiveCatalog(raw_dir / "TBX11K.zip")
    inventory = catalog.inspect()
    nlm_inventories = {
        site: summary.to_dict()
        for site, summary in NLMChestXrayCatalog(raw_dir).inspect().items()
    }
    roi_reader = ROIAnnotationCsvReader()
    roi_summaries = {}
    for site, filename in (
        ("Montgomery", Path("montgomery") / "montgomery_consensus_roi.csv"),
        ("Shenzhen", Path("shenzhen") / "shenzhen_consensus_roi.csv"),
    ):
        annotations = roi_reader.read(raw_dir / filename, site=site)
        roi_summaries[site] = asdict(roi_reader.summarize(site, annotations))

    print(
        json.dumps(
            {
                "tbx11k": inventory.to_dict(),
                "nlm_datasets": nlm_inventories,
                "roi_annotations": roi_summaries,
                "label_policy": {
                    "positive": "at_least_one_ActiveTuberculosis_lesion_box",
                    "excluded_tb_classes": ["obsolete_tb", "unresolved_tb"],
                    "negative_spectra": {
                        "healthy_only": ["healthy"],
                        "healthy_plus_non_tb_disease": ["healthy", "non_tb_disease"],
                    },
                    "roi_tables_are_image_level_labels": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _preprocess_data(
    project_root: Path,
    output_dir: Path | None,
    image_size: int,
    resize_policy: str,
    window_all_grayscale: bool,
) -> int:
    raw_dir = project_root / "data" / "raw"
    destination = output_dir or project_root / "data" / "processed" / "primary"
    if not destination.is_absolute():
        destination = project_root / destination
    config = CXRPreprocessingConfig(
        image_size=image_size,
        resize_policy=resize_policy,
        window_all_grayscale=window_all_grayscale,
    )
    summary = PrimaryCohortPreprocessor(raw_dir, destination, config).run()
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tb-certify",
        description="Research tools for TB CXR sensitivity certification.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "inspect-data",
        help="summarize public datasets, source labels, and ROI annotation tables",
    )
    inspect_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="repository root containing data/raw/",
    )
    preprocess_parser = subparsers.add_parser(
        "preprocess-data",
        help="build the deterministic 224-pixel primary image cache and manifest",
    )
    preprocess_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="repository root containing data/raw/",
    )
    preprocess_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="cache destination (default: data/processed/primary)",
    )
    preprocess_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="square output size in pixels (default: 224)",
    )
    preprocess_parser.add_argument(
        "--resize-policy",
        choices=("square_lanczos", "letterbox_lanczos"),
        default="square_lanczos",
        help="resize by square warp or aspect-preserving letterbox",
    )
    preprocess_parser.add_argument(
        "--window-all-grayscale",
        action="store_true",
        help="apply configured percentile windowing to 8-bit grayscale inputs too",
    )
    embedding_parser = subparsers.add_parser(
        "extract-embeddings",
        help="extract frozen encoder features into a revision-tagged NumPy cache",
    )
    embedding_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="repository root containing configs/ and data/processed/",
    )
    embedding_parser.add_argument(
        "--encoder",
        choices=("general", "chest_xray"),
        required=True,
        help="encoder key in configs/model.yaml",
    )
    embedding_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="inference batch size (default: selected by encoder/device)",
    )
    embedding_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device (default: use CUDA when available)",
    )
    embedding_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: artifacts/embeddings/<encoder>-<revision>)",
    )
    embedding_parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="preprocessed cache directory (default: encoder's standard cache)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    project_root = arguments.project_root.resolve()
    try:
        if arguments.command == "inspect-data":
            return _inspect_data(project_root)
        if arguments.command == "preprocess-data":
            return _preprocess_data(
                project_root,
                arguments.output_dir,
                arguments.image_size,
                arguments.resize_policy,
                arguments.window_all_grayscale,
            )
        if arguments.command == "extract-embeddings":
            from tb_certify.models.embeddings import extract_from_project

            summary = extract_from_project(
                project_root=project_root,
                encoder_key=arguments.encoder,
                batch_size=arguments.batch_size,
                device_name=arguments.device,
                output_dir=arguments.output_dir,
                cache_root=arguments.cache_root,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        parser.error(f"unknown command: {arguments.command}")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2
