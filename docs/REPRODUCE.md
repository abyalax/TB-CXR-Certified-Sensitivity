# Reproduce the experiments

This guide rebuilds the preprocessing caches, embeddings, model heads, calibration/evaluation outputs, audits, and figures for the current study. Run commands from the repository root in PowerShell. The commands use the pinned settings and seeds from this checkout.

## 1. Requirements and input data

Install `uv`, then install the locked project dependencies, including the optional vision stack:

```powershell
uv sync --extra vision
```

The source datasets are not included in the repository. Put them in the locations and directory structure described in [`../data/README.md`](../data/README.md). `download_dataset.ps1` downloads TBX11K; obtain Montgomery and Shenzhen from their NLM distribution pages and place them under `data/raw/montgomery/` and `data/raw/shenzhen/`. Review source terms before using or redistributing data. The commands below require the complete input files, including TBX11K.zip and the ROI CSVs.

Check the data inventory and label/ROI summaries:

```powershell
uv run tb-certify inspect-data --project-root .
```

For the three study datasets, one image represents one patient. The image-based split manifests are therefore patient-level. Generated results record `patient_level_claim: true`; PAC validity still depends on assumptions about representative sampling, fixed scores, accurate reference labels, and stationarity.

## 2. Build deterministic image caches

Build the standard cache for ResNet50-A1 and the separate 518-pixel cache for RAD-DINO:

```powershell
uv run tb-certify preprocess-data --project-root .
uv run tb-certify preprocess-data --project-root . --output-dir data/processed/rad_dino_518 --image-size 518
```

Expected outputs are `manifest.csv`, `preprocessing.json`, `hash_audit.json`, and cached PNGs under `data/processed/primary/` and `data/processed/rad_dino_518/`. Keep each cache aligned with its encoder input resolution.

## 3. Get the pinned RAD-DINO checkpoint and extract embeddings

Download the exact RAD-DINO revision referenced by `configs/model.yaml`, then extract both frozen encoders:

```powershell
uv run --extra vision hf download microsoft/rad-dino config.json preprocessor_config.json model.safetensors --revision 110cbc18d5133582e320b43d53bf5c44e410c936 --local-dir artifacts/checkpoints/rad-dino-110cbc18
uv run --extra vision tb-certify extract-embeddings --project-root . --encoder general
uv run --extra vision tb-certify extract-embeddings --project-root . --encoder chest_xray
```

The default output folders are `artifacts/embeddings/general-767268603ca0/` and `artifacts/embeddings/chest_xray-110cbc18d513/`. Each contains the embedding matrix, ordered manifest copy, and provenance metadata. Extraction refuses to write into a nonempty output folder; choose a fresh `--output-dir` or move the previous generated folder before rerunning. `--device auto` chooses CUDA when available; pass `--device cpu` to use CPU, understanding that it will take longer.

## 4. Primary and secondary site-holdout studies

Each study trains logistic and MLP heads on source-site embeddings, selects an empirical source-validation threshold, runs Tier A calibration/evaluation with 1,000 repetitions, and records a CRC/W-CRC/IW-LTT comparison at unit density weights in each head's `results.json`. Risk-control uses the full target-positive calibration pool and an independent target evaluation holdout. IW-LTT uses 25 unique source-validation score quantiles fixed independently of target calibration. Unit weights encode the exchangeable baseline; they do not adjust for site shift. The runner also writes its split manifest under `data/splits/`.

Run the primary TBX11K holdout and secondary Shenzhen holdout for both encoders:

```powershell
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513 --target-site TBX11K --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/general-767268603ca0 --output-dir experiments/runs/primary_tbx11k_loso/general-767268603ca0 --target-site TBX11K --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/secondary_shenzhen_loso/chest_xray-110cbc18d513 --target-site Shenzhen --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/general-767268603ca0 --output-dir experiments/runs/secondary_shenzhen_loso/general-767268603ca0 --target-site Shenzhen --repetitions 1000 --seed 42
```

Outputs are `results.json` files in each run folder. The secondary Shenzhen protocol performs a stratified 50/50 patient split, with one image per patient.

## 4a. CRC-family baseline comparison

The primary RAD-DINO TBX11K comparison is reproduced with a separate output folder so existing run results remain intact:

```powershell
uv run python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/risk_control_comparison --target-site TBX11K --repetitions 1000 --seed 42
```

Run the secondary Shenzhen holdout with the same encoder as follows:

```powershell
uv run python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/secondary_shenzhen_loso/chest_xray-110cbc18d513/risk_control_comparison --target-site Shenzhen --repetitions 1000 --seed 42
```

The output records CRC, unit-weight W-CRC, and unit-weight IW-LTT certificates and holdout metrics. This is the exchangeable baseline; shift-adjusted importance weighting is unavailable without a justified positive-class density ratio and upper bound.

## 5. Selective-verification simulations

These are additional primary TBX11K studies. The first evaluates the four-bound Clopper-Pearson procedure; the second also evaluates the linear IPW-Hoeffding comparator. Both use 100 selective-policy repetitions and 1,000 Tier A repetitions:

```powershell
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_8_selective --target-site TBX11K --repetitions 1000 --selective-repetitions 100 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_8_p1_linear_selective --target-site TBX11K --repetitions 1000 --selective-repetitions 100 --seed 42
```

## 6. Random-split control and lesion-size analysis

The site-mixed random-split control is patient-level because each image represents one patient. It is a negative control for site shift, not a transportability estimate. The lesion-size analysis describes only the annotated Shenzhen TB-positive patients and does not measure clinical severity.

```powershell
uv run --extra vision python scripts/run_random_split_control.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/random_image_control/chest_xray-110cbc18d513 --repetitions 1000 --seed 42
uv run --extra vision python scripts/analyze_lesion_size.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output experiments/runs/secondary_shenzhen_loso/chest_xray-110cbc18d513/p0_9_lesion_size/results.json
```

The random-split runner writes `data/splits/random_image_control_seed42.csv`.

## 7. Dataset-origin and RAD-DINO pretraining-overlap audits

The origin probe measures how much site information can be decoded from the embeddings. Download the pinned upstream training-image manifest before running the metadata-overlap audit:

```powershell
uv run --extra vision python scripts/probe_dataset_origin.py --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_4_origin_probe.json --seed 42 --folds 5
uv run --extra vision hf download microsoft/rad-dino training_images.csv --revision 110cbc18d5133582e320b43d53bf5c44e410c936 --local-dir artifacts/audits/rad-dino-110cbc18
uv run --extra vision python scripts/audit_rad_dino_training_overlap.py artifacts/audits/rad-dino-110cbc18/training_images.csv artifacts/embeddings/chest_xray-110cbc18d513/manifest.csv --output artifacts/audits/rad_dino_training_overlap_110cbc18.json
```
 
The overlap script checks metadata identifiers/names. It cannot check pixel-level duplicates because upstream training image pixels or pixel hashes are unavailable.

## 8. LoRA fine-tuning ablation (optional; CUDA recommended)

This trains a rank-8 LoRA adapter using the Montgomery and Shenzhen source cohort, then evaluates on TBX11K. It is separate from the frozen-embedding main study and may require a GPU with adequate memory.

```powershell
uv run --extra vision python scripts/run_lora_ablation.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p1_2_lora_final --epochs 3 --batch-size 1 --gradient-accumulation 4 --repetitions 1000 --seed 42
```

The runner writes the adapter, `classifier_head.pt`, `results.json`, and `data/splits/lora_source_seed42.csv`.

## 9. Render figures and run code checks

Regenerate the aggregate results thumbnail and workflow diagram from run JSON files:

```powershell
uv run --extra vision python scripts/render_result_thumbnails.py --output-dir paper/figures
```

Run the project tests and lint checks:

```powershell
uv run pytest
uv run ruff check src scripts
```

## What is and is not regenerated

The experiment runners generate the scenario-specific split manifests as they run; there is no separate split-generation CLI. Existing manifests in `data/splits/` record the current seed-42 assignments. Preprocessing, embedding extraction, model fitting, Tier A and CRC-family calibration/evaluation, LoRA, audits, and aggregate SVGs are covered above. Valid importance-weighted shift analysis requires externally justified positive-class density ratios and a known upper bound; the current study does not provide them.

The paper sources are `paper/imrad.md` and `paper/main.typ`. No paper compilation command is specified here because the checkout does not contain the Typst executable or its `juti` template dependency. Installing those external paper-build dependencies is required to compile `paper/main.typ`; the runnable analysis pipeline does not depend on them.

Dataset downloads require network access and acceptance of the source dataset terms. Model downloads require access to the pinned Hugging Face revision. Repeated bootstrap/resampling runs are deterministic from their recorded seed, subject to matching dependency versions, device behavior, and input file contents.
