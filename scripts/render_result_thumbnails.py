"""Render compact SVG previews directly from the latest experiment JSON files."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _text(x: int, y: int, value: str, size: int = 15, color: str = "#17233c") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">'
        f"{html.escape(value)}</text>"
    )


def _bar(x: int, y: int, value: float, label: str, color: str, width: int = 260) -> str:
    value_width = max(0, min(width, round(width * value)))
    return (
        _text(x, y + 15, label, 13) + f'<rect x="{x + 160}" y="{y}" width="{width}" '
        'height="16" rx="4" fill="#e7ebf2"/>'
        + f'<rect x="{x + 160}" y="{y}" width="{value_width}" '
        f'height="16" rx="4" fill="{color}"/>'
        + _text(x + 430, y + 14, f"{value:.1%}", 13, color)
    )


def _panel(
    x: int,
    y: int,
    title: str,
    rows: list[tuple[str, float, str]],
    height: int = 370,
) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="500" height="{height}" rx="14" '
        'fill="#ffffff" stroke="#d9deea"/>',
        _text(x + 20, y + 32, title, 18),
        _text(x + 180, y + 58, "proportion", 11, "#65718a"),
    ]
    for index, (label, value, color) in enumerate(rows):
        parts.append(_bar(x + 20, y + 72 + index * 39, value, label, color))
    return "\n".join(parts)


def render_results(output: Path) -> None:
    primary = _json(
        "experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/results.json"
    )
    random = _json(
        "experiments/runs/random_image_control/chest_xray-110cbc18d513/results.json"
    )
    lora = _json(
        "experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p1_2_lora_final/results.json"
    )
    selective = _json(
        "experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_8_selective/results.json"
    )
    roi = _json(
        "experiments/runs/secondary_shenzhen_loso/chest_xray-110cbc18d513/p0_9_lesion_size/results.json"
    )
    rows: list[tuple[str, float, str]] = []
    for head in ("logistic", "mlp"):
        metrics = primary["heads"][head]["target_b0_metrics"]
        rows.append((f"LOSO {head} sensitivity", metrics["sensitivity"], "#3976c5"))
        rows.append((f"LOSO {head} specificity", metrics["specificity"], "#7656bd"))
        metrics = random["heads"][head]["evaluation_b0_metrics"]
        rows.append((f"Mixed {head} sensitivity", metrics["sensitivity"], "#3976c5"))
        rows.append((f"Mixed {head} specificity", metrics["specificity"], "#7656bd"))
    rows.append(
        (
            "LoRA sensitivity",
            lora["target_b0_metrics"]["sensitivity"],
            "#3976c5",
        )
    )
    rows.append(
        (
            "LoRA specificity",
            lora["target_b0_metrics"]["specificity"],
            "#7656bd",
        )
    )
    panel_a = _panel(
        24, 72, "B0: source threshold on target evaluation", rows, height=490
    )

    rows = []
    for head in ("logistic", "mlp"):
        summaries = selective["heads"][head]["tier_a_exploratory"]["summaries"]
        for size in (29, 200):
            summary = next(
                item for item in summaries if item["requested_positive_count"] == size
            )
            rows.append(
                (
                    f"{head} n+={size} sensitivity",
                    summary["mean_evaluation_sensitivity"],
                    "#3976c5",
                )
            )
            rows.append(
                (
                    f"{head} n+={size} specificity",
                    summary["mean_evaluation_specificity"],
                    "#7656bd",
                )
            )
    lora_summaries = lora["tier_a_exploratory"]["summaries"]
    for size in (29, 200):
        summary = next(
            item for item in lora_summaries if item["requested_positive_count"] == size
        )
        rows.extend(
            [
                (
                    f"LoRA n+={size} sensitivity",
                    summary["mean_evaluation_sensitivity"],
                    "#3976c5",
                ),
                (
                    f"LoRA n+={size} specificity",
                    summary["mean_evaluation_specificity"],
                    "#7656bd",
                ),
            ]
        )
    panel_b = _panel(
        548,
        72,
        "Tier A: mean sensitivity / specificity",
        rows,
        height=560,
    )

    rows = []
    for q in (0.10, 0.25):
        entry = next(
            item
            for item in selective["heads"]["logistic"][
                "selective_verification_exploratory"
            ]["summaries"]
            if item["low_score_probability"] == q
        )
        naive = entry["naive_tier_a"]
        vb = entry["verification_bias_certificate"]
        rows.extend(
            [
                (f"q={q:.2f} naive issue", naive["issue_rate"], "#df8b2b"),
                (f"q={q:.2f} VB issue", vb["issue_rate"], "#3976c5"),
                (f"q={q:.2f} naive violation", naive["violation_rate"], "#c44949"),
                (f"q={q:.2f} VB violation", vb["violation_rate"], "#4e9b71"),
            ]
        )
    panel_c = _panel(24, 650, "Selective verification (logistic audit)", rows)

    rows = []
    for head in ("logistic", "mlp"):
        for stratum in ("small", "medium", "large"):
            value = roi["heads"][head]["by_lesion_size"][stratum]["sensitivity"]
            rows.append(
                (
                    f"{head} {stratum} ROI",
                    value,
                    "#3976c5" if head == "logistic" else "#7656bd",
                )
            )
    panel_d = _panel(548, 650, "Shenzhen positive-only ROI subset (n=68)", rows)

    svg = "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1050"',
            ' viewBox="0 0 1080 1050">',
            '<rect width="1080" height="1050" fill="#f4f6fa"/>',
            '<text x="24" y="36" font-family="Arial, sans-serif"',
            ' font-size="23" font-weight="bold" fill="#17233c">',
            "Latest aggregate audit results - image-level</text>",
            '<text x="24" y="58" font-family="Arial, sans-serif"',
            ' font-size="13" fill="#65718a">Generated from experiment JSON;',
            " descriptive only, patient independence unverified.</text>",
            '<g font-family="Arial, sans-serif">',
            panel_a,
            panel_b,
            panel_c,
            panel_d,
            "</g>",
            "</svg>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")


def render_workflow(output: Path) -> None:
    boxes = [
        (30, 90, 220, 100, "SOURCE SITES", "Train frozen/LoRA head"),
        (310, 90, 220, 100, "SOURCE VALIDATION", "Fix score threshold / model"),
        (590, 90, 220, 100, "TARGET POSITIVES", "Draw calibration sample"),
        (870, 90, 220, 100, "CERTIFY OR REFUSE", "Exact binomial order statistic"),
        (590, 285, 220, 100, "TARGET HOLDOUT", "Estimate sensitivity / specificity"),
        (
            310,
            285,
            220,
            100,
            "SELECTIVE LABELS",
            "Known verification policy in simulation",
        ),
        (30, 285, 220, 100, "ASSUMPTIONS", "Patient i.i.d., fixed model, stationarity"),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="450" '
        'viewBox="0 0 1120 450">',
        '<rect width="1120" height="450" fill="#f4f6fa"/>',
        '<text x="30" y="42" font-family="Arial" font-size="24" '
        'font-weight="bold" fill="#17233c">Study workflow and evidence boundary</text>',
    ]
    for x1, y1, x2, y2 in (
        (250, 140, 310, 140),
        (530, 140, 590, 140),
        (810, 140, 870, 140),
        (700, 190, 700, 285),
        (530, 335, 590, 335),
        (250, 335, 310, 335),
    ):
        parts.append(
            f'<path d="M{x1} {y1} L{x2} {y2}" stroke="#65718a" '
            'stroke-width="3" marker-end="url(#arrow)"/>'
        )
    for x, y, width, height, title, subtitle in boxes:
        parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            'rx="12" fill="#ffffff" stroke="#3976c5" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x + 14}" y="{y + 35}" font-family="Arial" '
            f'font-size="15" font-weight="bold" fill="#17233c">'
            f"{html.escape(title)}</text>"
        )
        parts.append(
            f'<text x="{x + 14}" y="{y + 65}" font-family="Arial" '
            f'font-size="13" fill="#65718a">{html.escape(subtitle)}</text>'
        )
    parts.append(
        '<defs><marker id="arrow" markerWidth="10" height="10" refX="8" '
        'refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" '
        'fill="#65718a"/></marker></defs>'
    )
    parts.append(
        '<text x="30" y="430" font-family="Arial" font-size="13" '
        'fill="#8a3b3b">All current public-data splits are image-level; '
        "patient-level validity is not established.</text>"
    )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "figures")
    args = parser.parse_args()
    render_results(args.output_dir / "latest_results.svg")
    render_workflow(args.output_dir / "study_workflow.svg")
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
