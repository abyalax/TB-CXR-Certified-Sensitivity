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

The manifests split image records. Patient linkage is unverified, so these are image-level splits and generated results must not claim patient-level validation. PAC interpretation still depends on representative sampling, fixed scores, accurate reference labels, and stationarity.

## 2. Build deterministic image caches

Build the standard cache for ResNet50-A1 and the separate 518-pixel cache for RAD-DINO:

```powershell
uv run tb-certify preprocess-data --project-root .
uv run tb-certify preprocess-data --project-root . --output-dir data/processed/rad_dino_518 --image-size 518
```

Expected outputs are `manifest.csv`, `preprocessing.json`, `hash_audit.json`, and cached PNGs under `data/processed/primary/` and `data/processed/rad_dino_518/`. Keep each cache aligned with its encoder input resolution.

The exploratory preprocessing audit can rebuild comparable caches with global grayscale percentile windowing or aspect-preserving letterboxing. Run the additional variants, then extract each encoder into a fresh output directory with the matching `--cache-root`:

```powershell
uv run tb-certify preprocess-data --project-root . --output-dir data/processed/ablation/window-all-224 --image-size 224 --window-all-grayscale
uv run tb-certify preprocess-data --project-root . --output-dir data/processed/ablation/letterbox-224 --image-size 224 --resize-policy letterbox_lanczos
uv run tb-certify preprocess-data --project-root . --output-dir data/processed/ablation/window-all-518 --image-size 518 --window-all-grayscale
uv run tb-certify preprocess-data --project-root . --output-dir data/processed/ablation/letterbox-518 --image-size 518 --resize-policy letterbox_lanczos
uv run --extra vision tb-certify extract-embeddings --project-root . --encoder general --cache-root data/processed/ablation/window-all-224 --output-dir artifacts/embeddings/ablation/window-all-224/general
uv run --extra vision tb-certify extract-embeddings --project-root . --encoder general --cache-root data/processed/ablation/letterbox-224 --output-dir artifacts/embeddings/ablation/letterbox-224/general
uv run --extra vision tb-certify extract-embeddings --project-root . --encoder chest_xray --cache-root data/processed/ablation/window-all-518 --output-dir artifacts/embeddings/ablation/window-all-518/chest_xray
uv run --extra vision tb-certify extract-embeddings --project-root . --encoder chest_xray --cache-root data/processed/ablation/letterbox-518 --output-dir artifacts/embeddings/ablation/letterbox-518/chest_xray
```

Run `scripts/probe_dataset_origin.py` on each embedding output with `--identity-audit artifacts/audits/image_identity.json` to exclude exact-pixel aliases. These changes are sensitivity analyses; they do not independently establish or remove causal shortcut use.

The four ablation origin probes are reproduced with the following commands:

```powershell
uv run python scripts/probe_dataset_origin.py --embedding-dir artifacts/embeddings/ablation/window-all-224/general --identity-audit artifacts/audits/image_identity.json --output experiments/runs/preprocessing_ablation/general-window-all-224-origin.json --seed 42 --folds 5
uv run python scripts/probe_dataset_origin.py --embedding-dir artifacts/embeddings/ablation/letterbox-224/general --identity-audit artifacts/audits/image_identity.json --output experiments/runs/preprocessing_ablation/general-letterbox-224-origin.json --seed 42 --folds 5
uv run python scripts/probe_dataset_origin.py --embedding-dir artifacts/embeddings/ablation/window-all-518/chest_xray --identity-audit artifacts/audits/image_identity.json --output experiments/runs/preprocessing_ablation/chest-xray-window-all-518-origin.json --seed 42 --folds 5
uv run python scripts/probe_dataset_origin.py --embedding-dir artifacts/embeddings/ablation/letterbox-518/chest_xray --identity-audit artifacts/audits/image_identity.json --output experiments/runs/preprocessing_ablation/chest-xray-letterbox-518-origin.json --seed 42 --folds 5
```

The completed probes exclude 126 exact-pixel aliases. ResNet50-A1 window-all-224 and letterbox-224 achieved 98.4% and 98.7% balanced accuracy; RAD-DINO window-all-518 and letterbox-518 achieved 98.3% and 99.2%. These high values show that site information remains readily decodable after these preprocessing changes; they do not identify its cause or prove TB-label shortcutting. Results are recorded under `experiments/runs/preprocessing_ablation/`.

## 3. Get the pinned RAD-DINO checkpoint and extract embeddings

Download the exact RAD-DINO revision referenced by `configs/model.yaml`, then extract both frozen encoders:

```powershell
uv run --extra vision hf download microsoft/rad-dino config.json preprocessor_config.json model.safetensors --revision 110cbc18d5133582e320b43d53bf5c44e410c936 --local-dir artifacts/checkpoints/rad-dino-110cbc18
uv run --extra vision tb-certify extract-embeddings --project-root . --encoder general
uv run --extra vision tb-certify extract-embeddings --project-root . --encoder chest_xray
```

The default output folders are `artifacts/embeddings/general-767268603ca0/` and `artifacts/embeddings/chest_xray-110cbc18d513/`. Each contains the embedding matrix, ordered manifest copy, and provenance metadata. Extraction refuses to write into a nonempty output folder; choose a fresh `--output-dir` or move the previous generated folder before rerunning. `--device auto` chooses CUDA when available; pass `--device cpu` to use CPU, understanding that it will take longer.

## 4. Primary and secondary site-holdout studies

Each study trains logistic and MLP heads on source-site embeddings, selects an empirical source-validation threshold, runs Tier A calibration/evaluation with 1,000 repetitions, and records CRC/IW-LTT comparisons at unit density weights in each head's `results.json`. Unit-weight W-CRC is algebraically identical to CRC and is not a separate result. Risk-control uses the full target-positive calibration pool and an independent target evaluation holdout. IW-LTT uses 25 unique source-validation score quantiles fixed independently of target calibration. Unit weights encode the exchangeable baseline; they do not adjust for site shift. The runner also writes its split manifest under `data/splits/`.

Run the primary TBX11K holdout and secondary Shenzhen holdout for both encoders:

```powershell
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513 --target-site TBX11K --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/general-767268603ca0 --output-dir experiments/runs/primary_tbx11k_loso/general-767268603ca0 --target-site TBX11K --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/secondary_shenzhen_loso/chest_xray-110cbc18d513 --target-site Shenzhen --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/general-767268603ca0 --output-dir experiments/runs/secondary_shenzhen_loso/general-767268603ca0 --target-site Shenzhen --repetitions 1000 --seed 42
```

Outputs are `results.json` files in each run folder. The secondary Shenzhen protocol performs a stratified 50/50 image split; patient linkage is unverified.

## 4a. CRC-family baseline comparison

The primary RAD-DINO TBX11K comparison is reproduced with a separate output folder so existing run results remain intact:

```powershell
uv run python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/risk_control_comparison --target-site TBX11K --repetitions 1000 --seed 42
```

Run the secondary Shenzhen holdout with the same encoder as follows:

```powershell
uv run python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/secondary_shenzhen_loso/chest_xray-110cbc18d513/risk_control_comparison --target-site Shenzhen --repetitions 1000 --seed 42
```

The output records CRC and unit-weight IW-LTT certificates and holdout metrics. W-CRC with all weights equal to one is identical to CRC, so it is omitted as a separate row. This is the exchangeable baseline; shift-adjusted importance weighting is unavailable without a justified positive-class density ratio and upper bound.

## 4b. Multi-seed and nested-resampling audits

The robustness runner performs ten source-head seeds, 1,000 repeated target-positive calibration/holdout splits, the expanded positive calibration-size grid, CRC comparisons, a capped 20-repetition IW-LTT diagnostic, domain-classifier density-ratio diagnostics, a within-site random control, and a 2,000-repetition WSR null simulation. Each IW-LTT run uses ten random orderings, so its Monte Carlo results are preliminary. The domain odds are diagnostics only and must not be passed to a guaranteed weighted procedure.

```powershell
uv run python -m scripts.run_robustness_audits --project-root . --embedding-dir artifacts/embeddings/general-767268603ca0 --output-dir experiments/runs/robustness_audits/general-767268603ca0 --repetitions 1000 --ltt-repetitions 20 --type-i-repetitions 2000 --seeds 42 43 44 45 46 47 48 49 50 51
```

The runner writes `results.json` and the within-site split manifest. All outcomes are image-level; patient linkage is unverified.

The raw-pixel site-origin comparator uses native dimensions and aspect ratio plus grayscale intensity summaries, and excludes exact-pixel duplicate aliases:

```powershell
uv run python -m scripts.probe_pixel_site_features --project-root . --manifest data/processed/primary/manifest.csv --output experiments/runs/primary_tbx11k_loso/pixel_site_probe.json
```

The RAD-DINO and ResNet50-A1 audits are both complete. The 10-seed B0 results vary across heads and targets; for example, RAD-DINO TBX11K sensitivity averaged 92.6% (logistic) and 94.6% (MLP), while ResNet50-A1 averaged 87.6% and 87.1%. Across both encoders, the 1,000-repetition nested image-level target resampling shows that empirical Tier A violation rates are not uniformly within the nominal 5% rate on the finite pools; these are image-level stress tests and do not establish patient-i.i.d. PAC validity. See each `results.json` for the full grid and Monte Carlo intervals.

The WSR null check observed 50/2,000 false rejections (2.5%; exact 95% interval 1.86%–3.28%) under iid Bernoulli null losses. Cross-fitted site odds and their effective sample sizes are diagnostics only, not justified density ratios or valid weighting inputs. Both interpretations are recorded in the run JSON.

~~~
powershell
uv run python -m scripts.run_robustness_audits --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/robustness_audits/chest_xray-110cbc18d513 --repetitions 1000 --ltt-repetitions 20 --type-i-repetitions 2000 --seeds 42 43 44 45 46 47 48 49 50 51
~~~

## 5. Selective-verification simulations

These are additional primary TBX11K studies. The first evaluates the four-bound Clopper-Pearson procedure; the second also evaluates the linear IPW-Hoeffding comparator. Both use 100 selective-policy repetitions and 1,000 Tier A repetitions:

```powershell
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_8_selective --target-site TBX11K --repetitions 1000 --selective-repetitions 100 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_8_p1_linear_selective --target-site TBX11K --repetitions 1000 --selective-repetitions 100 --seed 42
```

The budget and prevalence sweep evaluates screening N in {1,000, 2,000, 4,000, 6,496}, prevalence in {0.05, 0.10, 0.20, 0.40}, and low-score verification probability q in {0.10, 0.25, 0.50, 1.00}. It resamples calibration and evaluation scores independently from the official TBX11K train split's class-conditional empirical score pools after exact-pixel alias exclusion. This is a simulation under that bootstrap model, not an external validation cohort:

~~~
powershell
uv run python -m scripts.run_selective_budget_sweep --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_8_budget_sweep --repetitions 100 --evaluation-count 2000 --seed 42
~~~

The label sensitivity runner embeds the 140 official TBX11K obsolete/unresolved images and compares mappings that assign all of them positive versus negative labels. Run it once per encoder:

~~~
powershell
uv run --extra vision python -m scripts.run_ambiguous_label_sensitivity --project-root . --embedding-dir artifacts/embeddings/general-767268603ca0 --encoder general --output-dir experiments/runs/label_sensitivity/general-767268603ca0 --repetitions 1000 --seed 42
uv run --extra vision python -m scripts.run_ambiguous_label_sensitivity --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --encoder chest_xray --output-dir experiments/runs/label_sensitivity/chest_xray-110cbc18d513 --repetitions 1000 --seed 42
~~~

## 6. Random-split control and lesion-size analysis

The site-mixed random-split control is image-level; patient linkage is unverified. It is a negative control for site shift, not a transportability estimate. The lesion-size analysis describes only the annotated Shenzhen TB-positive images and does not measure clinical severity.

```powershell
uv run --extra vision python scripts/run_random_split_control.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/random_image_control/chest_xray-110cbc18d513 --repetitions 1000 --seed 42
uv run --extra vision python scripts/analyze_lesion_size.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output experiments/runs/secondary_shenzhen_loso/chest_xray-110cbc18d513/p0_9_lesion_size/results.json
```

The random-split runner writes `data/splits/random_image_control_seed42.csv`.

## 7. Dataset-origin and RAD-DINO pretraining-overlap audits

The origin probe measures how much site information can be decoded from the embeddings. Download the pinned upstream training-image manifest before running the metadata-overlap audit:

```powershell
uv run python scripts/probe_dataset_origin.py --embedding-dir artifacts/embeddings/general-767268603ca0 --identity-audit artifacts/audits/image_identity.json --output experiments/runs/primary_tbx11k_loso/general-767268603ca0/p0_4_origin_probe_deduped.json --seed 42 --folds 5
uv run python scripts/probe_dataset_origin.py --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --identity-audit artifacts/audits/image_identity.json --output experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_4_origin_probe_deduped.json --seed 42 --folds 5
uv run --extra vision hf download microsoft/rad-dino training_images.csv --revision 110cbc18d5133582e320b43d53bf5c44e410c936 --local-dir artifacts/audits/rad-dino-110cbc18
uv run --extra vision python scripts/audit_rad_dino_training_overlap.py artifacts/audits/rad-dino-110cbc18/training_images.csv artifacts/embeddings/chest_xray-110cbc18d513/manifest.csv --output artifacts/audits/rad_dino_training_overlap_110cbc18.json
```
 
The overlap script checks metadata identifiers/names. It cannot check pixel-level duplicates because upstream training image pixels or pixel hashes are unavailable.

## 8. LoRA fine-tuning ablation (optional; CUDA recommended)

This trains a rank-8 LoRA adapter using the Montgomery and Shenzhen source cohort, then evaluates on TBX11K. It is separate from the frozen-embedding main study and may require a GPU with adequate memory.

```powershell
uv run --extra vision python scripts/run_lora_ablation.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p1_2_lora_final --epochs 3 --batch-size 1 --gradient-accumulation 4 --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_lora_ablation.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p1_2_lora_seed43 --epochs 3 --batch-size 1 --gradient-accumulation 4 --repetitions 1000 --seed 43
```

Each run directory writes the adapter, `classifier_head.pt`, and `results.json`; its source split manifest is stored separately under `data/splits/`. Seed 43 is the independent-seed sensitivity run.

At $n^+=200$, seed 42 had 1.2% observed violations and seed 43 had 17.7% on the same finite TBX11K image holdout pool. This large run-to-run difference is descriptive image-level evidence; it neither validates nor disproves patient-population PAC coverage.

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
