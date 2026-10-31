# Research Documentation: TB Sensitivity Certification on Chest X-Ray (CXR) Images

## Workflow Overview

Penelitian ini membangun sistem klasifikasi TB dari citra rontgen dada (CXR) yang disertai ****jaminan sensitivitas**** (bukan sekadar skor klasifikasi biasa) saat dipakai di situs/data baru. Alurnya, secara garis besar: kumpulkan dataset mentah -> preprocessing & split (dua resolusi) -> ekstraksi embedding (dua encoder) -> audit data leakage -> latih head klasifikasi, threshold calibration, dan verifikasi selektif pada beberapa skenario evaluasi (LOSO, random-split control, ablasi LoRA) -> simpan hasil per run -> tulis ke paper format IMRaD/Typst.

All experiments use `seed=42` for reproducibility. Unless stated otherwise, commands are executed from the repository root in PowerShell.

---

## 1. Environment Setup and Raw Data

### 1.1 Install Dependencies

The project uses `uv` as its package manager. Install the project dependencies, including the optional vision stack, with:

```powershell
uv sync --extra vision
```

### 1.2 Datasets

The pipeline uses three source datasets, stored under `data/raw/`:

- ****Montgomery**** dan ****Shenzhen**** — masing-masing dengan `ClinicalReadings`, `CXR_png`, `ManualMask/Annotations` ganda, serta file ROI konsensus.
- ****TBX11K**** (`TBX11K.zip`).

Source datasets are not included in this repository. Download Montgomery and Shenzhen from the NLM distribution pages and place them in `data/raw/montgomery/` and `data/raw/shenzhen/`, following the directory structure described in `data/README.md`. Download TBX11K using the following script:

```powershell
./download_dataset.ps1
```

Review the usage and redistribution terms of each dataset before use or redistribution.

Patient linkage is unverified (see `artifacts/audits/image_identity.json`). The generated manifests split image records and do not establish patient-level separation.

### 1.3 Inspect the Dataset Inventory

Inspect the dataset inventory and review the label/ROI summaries before running preprocessing:

```powershell
uv run tb-certify inspect-data --project-root .
```
---

## 2. Image Preprocessing and Data Splitting

Raw images are processed into two cached variants under `data/processed/` because the two encoders require different input resolutions:

- ****`primary`**** — ukuran asli, used for encoder ResNet50-A1.
- ****`rad_dino_518`**** — khusus resolusi 518×518, used for RAD-DINO.

Each variant includes `manifest.csv`, `hash_audit.json`, and `preprocessing.json` for traceability.
Split (`data/splits/`) dibuat per skenario evaluasi: LOSO (**leave-one-site-out**) untuk primary (target TBX11K) dan secondary (target Shenzhen), source split, random control, serta split khusus sumber untuk ablasi LoRA. Seed dasar adalah 42, robustness memakai seed 42–51, dan LoRA memiliki run independen dengan seed 43.

### Run Pipeline

```powershell
uv run tb-certify preprocess-data --project-root .
uv run tb-certify preprocess-data --project-root . --output-dir data/processed/rad_dino_518 --image-size 518
```

Expected outputs include: `manifest.csv`, `preprocessing.json`, `hash_audit.json`, dan PNG hasil cache di bawah `data/processed/primary/` dan `data/processed/rad_dino_518/`. Setiap cache harus tetap selaras dengan resolusi input encoder-nya masing-masing.

Scenario-specific split manifests are generated automatically by the experiment runners (Section 6). No separate CLI command is provided for split generation. Existing manifests in `data/splits/` record the current assignments generated with `seed=42`.

---

## 3. Feature and Embedding Extraction

Dua encoder pralatih yang ****dibekukan**** (frozen, tidak di-fine-tune pada data TB di jalur eksperimen utama) used for mengekstrak embedding (`artifacts/embeddings/`):

- `chest_xray-110cbc18d513` — ****RAD-DINO****, input 518×518, CLS embedding 768 dimensi.
- `general-767268603ca0` — ****ResNet50-A1 ImageNet****, sebagai pembanding.

The RAD-DINO checkpoint is stored and its hash is verified before embedding extraction. The extraction implementation is located in `src/tb_certify/models/embeddings.py`; fine-tuning RAD-DINO dengan LoRA (ablation terpisah) ada di `scripts/run_lora_ablation.py` (dibahas di bagian 6.5).

### Run the Pipeline

Unduh revisi RAD-DINO yang dipin di `configs/model.yaml`, lalu ekstrak kedua encoder:

```powershell
uv run --extra vision hf download microsoft/rad-dino config.json preprocessor_config.json model.safetensors --revision 110cbc18d5133582e320b43d53bf5c44e410c936 --local-dir artifacts/checkpoints/rad-dino-110cbc18
uv run --extra vision tb-certify extract-embeddings --project-root . --encoder general
uv run --extra vision tb-certify extract-embeddings --project-root . --encoder chest_xray
```

The default output directories are: `artifacts/embeddings/general-767268603ca0/` dan `artifacts/embeddings/chest_xray-110cbc18d513/`, masing-masing berisi embedding matrix, salinan manifest terurut, dan metadata provenance (sidecar provenance ditulis dari `src/tb_certify/models/embeddings.py`, divalidasi lewat `src/tb_certify/models/provenance.py`).

Note: embedding extraction refuses to write to a non-empty output directory. Specify a new directory with `--output-dir` or move the existing directory before rerunning. `--device auto` selects CUDA when available, while `--device cpu` forces CPU execution (slower).

---

## 4. Data Leakage Audits

Karena RAD-DINO adalah model **pretrained** (bukan dilatih dari nol), the pipeline perlu memastikan tidak ada data leakage eval ke data training RAD-DINO sebelum embedding-nya dipercaya untuk eksperimen.

### 4.1 Site-Origin Probe on Embeddings

`scripts/probe_dataset_origin.py:28-85` mengukur seberapa banyak informasi domain (site origin) yang bisa dibaca dari embedding frozen — ****ini bukan prediksi TB, dan bukan bukti penyebab shortcut****, hanya pengukuran kebocoran informasi domain. Langkahnya: ambil embedding frozen dan label `site` dari manifest (`:30-42`), lalu `StandardScaler` + balanced Logistic Regression (`:43-56`) dengan Stratified 5-fold cross-validation (`:57-60`).

The latest RAD-DINO result is recorded in `experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_4_origin_probe_deduped.json`: after excluding 126 exact-pixel aliases, accuracy is 0.9934, balanced accuracy 0.9860, and macro OVR AUC 0.9992. This is a site-origin probe, not a TB classification metric or evidence of causal shortcutting.

### 4.2 RAD-DINO Pretraining-Overlap Audit

`audit_rad_dino_training_overlap.py` mengecek apakah evaluation data the pipeline overlap dengan data training RAD-DINO, berdasarkan metadata identifier/nama — ****bukan**** pixel-level duplicate check, karena pixel atau pixel hash dari data training upstream tidak tersedia.

### Run the Pipeline

```powershell
uv run python scripts/probe_dataset_origin.py --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --identity-audit artifacts/audits/image_identity.json --output experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_4_origin_probe_deduped.json --seed 42 --folds 5
uv run --extra vision hf download microsoft/rad-dino training_images.csv --revision 110cbc18d5133582e320b43d53bf5c44e410c936 --local-dir artifacts/audits/rad-dino-110cbc18
uv run --extra vision python scripts/audit_rad_dino_training_overlap.py artifacts/audits/rad-dino-110cbc18/training_images.csv artifacts/embeddings/chest_xray-110cbc18d513/manifest.csv --output artifacts/audits/rad_dino_training_overlap_110cbc18.json
```
---

## 5. Classification-Head Training, Threshold Calibration, and Selective Verification

After the embedding pipeline has been assessed for potential leakage, the classification head is trained on top of the embeddings without retraining the encoder. Its decision threshold is then calibrated and statistically certified.

- ****Encoder fitur****: RAD-DINO dan ResNet50-A1 dibekukan pada studi utama; ekstraksi di `src/tb_certify/models/embeddings.py`.
- ****Training head****: `src/tb_certify/training/source_head.py` melatih standardized Logistic Regression dari embedding data source site (MLP sebagai pembanding). Site target yang sedang diuji ****dikecualikan**** dari training head — ini classifier di atas embedding, bukan bagian dari encoder.
- ****Threshold baseline****: dipilih dari validasi source via `choose_source_threshold` di `src/tb_certify/training/source_head.py`; sifatnya empiris, ****bukan sertifikat****.
- ****Kalibrasi threshold****: `src/tb_certify/calibration/source_threshold.py` menangani threshold empiris; `src/tb_certify/calibration/tier_a.py` mengimplementasikan Tier A; `src/tb_certify/calibration/risk_control.py` mengimplementasikan marginal CRC, test-covariate W-CRC, dan IW-LTT. Runner melaporkan CRC dan IW-LTT berbobot satu pada holdout terpisah. W-CRC berbobot kalibrasi dan titik uji satu identik secara aljabar dengan CRC, sehingga tidak ditampilkan sebagai hasil terpisah. Bobot satu hanya mewakili kasus exchangeable dan bukan koreksi shift. These are statistical certification procedures (Clopper–Pearson bounds and order statistics), not machine-learning models.
- ****Selective verification (certify-or-refuse)****: `src/tb_certify/verification/selective.py` mengimplementasikan four-CP dan linear importance-weighted Hoeffding UCB. Bound IPW yang dihitung per threshold adalah:

```latex
V \cdot Y \cdot (\mathbb{1}\\{\text{score} < t\\} - \alpha)/\pi(\text{score})
```
Thresholds are evaluated sequentially. A threshold is accepted when its UCB is ≤ 0; otherwise, the certification procedure returns a refusal.

- ****Evaluasi****: `src/tb_certify/evaluation/selective_study.py` menjalankan selective verification dan mengevaluasi threshold pada data terpisah; `src/tb_certify/evaluation/tier_a_study.py` menjalankan studi Tier A; `src/tb_certify/evaluation/binomial.py` menghitung interval Clopper-Pearson; `src/tb_certify/evaluation/metrics.py` menghitung metrik threshold.
- ****Provenance****: `src/tb_certify/models/provenance.py` menyimpan/memvalidasi asal model dan data.

Each classification run follows this sequence: load the frozen embeddings (RAD-DINO or ResNet50-A1) and split manifest -> fit a Logistic Regression or MLP head on the training data -> select the empirical baseline threshold from source-site validation data for a target sensitivity of 0.90 -> compute metrics on the validation and target evaluation sets. Under LOSO, the target site is excluded from head training. Each run records the seed, cohort/split sizes, encoder metadata, threshold, and per-head results in `results.json`.

---

## 6. Running Experiments

All experiments are stored under `experiments/runs/`. Each run writes its results to `results.json` in a dedicated output directory. Additional artifacts, such as split manifests and model metadata, may be stored alongside the results. Visual summaries are generated separately (see Section 7).

### 6.1 Studi utama: `primary_tbx11k_loso` dan `secondary_shenzhen_loso`

Both studies train Logistic Regression/MLP heads on source-site embeddings, select an empirical threshold using source-site validation data, and run Tier A calibration/evaluation with 1,000 repetitions. Each runner also writes its split manifest to `data/splits/`.

- ****`primary_tbx11k_loso`****: **leave-one-site-out** dengan TBX11K sebagai target, site lain sebagai sumber. Dari data target, protokol memisahkan data kalibrasi positif dan evaluasi; menjalankan baseline threshold empiris serta studi Tier A pada berbagai ukuran kalibrasi/repetisi. Untuk TBX11K juga ada negative control (`healthy_only`, `healthy_plus_non_tb`). Selective verification eksploratori hanya dihitung jika `--selective-repetitions` diaktifkan (default 0).
- ****`secondary_shenzhen_loso`****: protokol LOSO serupa dengan Shenzhen sebagai target (stratified 50/50 image split; patient linkage unverified); head dilatih pada source site, lalu threshold baseline dan Tier A dievaluasi pada target.

```powershell
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513 --target-site TBX11K --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/general-767268603ca0 --output-dir experiments/runs/primary_tbx11k_loso/general-767268603ca0 --target-site TBX11K --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/secondary_shenzhen_loso/chest_xray-110cbc18d513 --target-site Shenzhen --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/general-767268603ca0 --output-dir experiments/runs/secondary_shenzhen_loso/general-767268603ca0 --target-site Shenzhen --repetitions 1000 --seed 42
```

### 6.2 Simulasi verifikasi selektif (`p0_8_selective`, `p0_8_p1_linear_selective`)

Dua studi tambahan di atas `primary_tbx11k_loso`: yang pertama mengevaluasi prosedur four-bound Clopper-Pearson; yang kedua menambah evaluasi linear IPW-Hoeffding comparator. Keduanya memakai 100 repetisi selective-policy dan 1.000 repetisi Tier A. Dibandingkan di sini: Tier A naif, four-CP verification-bias certificate, dan linear IPW-Hoeffding certificate. Keluaran utamanya adalah threshold yang lolos/ditolak, bound sertifikasi, dan perilaku refuse — ****bukan**** confusion matrix.

```powershell
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_8_selective --target-site TBX11K --repetitions 1000 --selective-repetitions 100 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_8_p1_linear_selective --target-site TBX11K --repetitions 1000 --selective-repetitions 100 --seed 42
```

### 6.3 Kontrol random split (`random_image_control`)

Kontrol dengan pembagian acak tingkat gambar, site tercampur; patient linkage tidak terverifikasi dan exact duplikat dikeluarkan sebelum split. Logistic Regression dan MLP dievaluasi pada split evaluasi; threshold berasal dari validasi, Tier A memakai split kalibrasi tersendiri. Tujuannya membandingkan dengan LOSO (bukan menguji transportabilitas lintas site) — ****negative control untuk site shift****, bukan estimasi transportabilitas.

```powershell
uv run --extra vision python scripts/run_random_split_control.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/random_image_control/chest_xray-110cbc18d513 --repetitions 1000 --seed 42
```

Runner ini menulis `data/splits/random_image_control_seed42.csv`.

### 6.4 Analisis ukuran lesi (`p0_9_lesion_size`)

Analisis sensitivitas threshold pada strata ukuran lesi, subset positive images Shenzhen yang beranotasi ROI. Ini analisis subgroup deskriptif — ****bukan**** training model klasifikasi baru, dan tidak mengukur keparahan klinis.

```powershell
uv run --extra vision python scripts/analyze_lesion_size.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output experiments/runs/secondary_shenzhen_loso/chest_xray-110cbc18d513/p0_9_lesion_size/results.json
```

### 6.5 Ablasi LoRA (`p1_2_lora`, `p1_2_lora_final`, `p1_2_lora_seed43`) — opsional, disarankan CUDA

This optional ablation fine-tunes RAD-DINO using rank-8 LoRA on the Montgomery and Shenzhen source cohorts, then evaluates the resulting classification head on TBX11K. It is separate from the primary frozen-embedding study and may require a GPU with sufficient memory.

```powershell
uv run --extra vision python scripts/run_lora_ablation.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p1_2_lora_final --epochs 3 --batch-size 1 --gradient-accumulation 4 --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_lora_ablation.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p1_2_lora_seed43 --epochs 3 --batch-size 1 --gradient-accumulation 4 --repetitions 1000 --seed 43
```

Runner menulis adapter, `classifier_head.pt`, `results.json`, dan manifest sumber per seed di `data/splits/`. Seed 43 adalah satu audit sensitivitas independen, bukan studi replikasi yang cukup untuk klaim multi-seed.

### 6.6 Metrik threshold / confusion matrix

For each evaluation threshold, `src/tb_certify/evaluation/metrics.py` menghitung TP, FN, TN, FP, sensitivity, dan specificity. Komponen ini diserialisasi pada `results.json` sebagai `source_validation_metrics`, `target_b0_metrics` (atau `evaluation_b0_metrics` di kontrol random), dan metrik negative control jika berlaku. The pipeline does not currently generate a separate confusion-matrix figure or table. These metrics are computed at the threshold specified by the evaluation protocol.

### 6.7 Skrip orkestrasi (`scripts/`)

| Script                                | Fungsi                                                                                                                             |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/run_primary_real_study.py`   | Menjalankan studi leave-site-out: melatih head dari source site, threshold calibration, mengevaluasi sertifikasi pada target site. |
| `scripts/run_random_split_control.py` | Menjalankan kontrol dengan split gambar acak yang mencampur site, untuk membandingkan hasil saat distribution shift tidak dijaga.  |
| `scripts/run_lora_ablation.py`        | Fine-tune RAD-DINO dengan LoRA pada data sumber, mengaudit performa head hasilnya pada TBX11K.                                     |
| `scripts/analyze_lesion_size.py`      | Menganalisis sensitivitas threshold menurut strata ukuran lesi pada subset positive images beranotasi ROI.                         |
| `scripts/probe_dataset_origin.py`     | Menguji seberapa akurat site origin dapat diprediksi dari embedding frozen via cross-validation.                                   |

---

## 7. Figures and Code Quality Checks

After all runs have generated their `results.json` files, regenerate the aggregate result thumbnails and workflow diagrams from the run JSON files:

```powershell
uv run --extra vision python scripts/render_result_thumbnails.py --output-dir paper/figures
```

Run the test suite and linter before using the results in subsequent analysis:

```powershell
uv run pytest
uv run ruff check src scripts
```

Unit tests are provided for the calibration and verification modules (including `test_risk_control.py` and `test_selective.py`) to validate the implementation before use in the experiments.

---

## 8. Paper Preparation (`paper/`)

- `imrad.md` is the active manuscript draft and is compiled into `main.typ` using Typst. A local `typst.exe` binary is stored in `.tools/`.
- `literature_search.md` dan `references.bib` untuk sitasi.
- The `SA-CTC for Tuberculosis/` directory contains an older or parallel draft, including `Literature Review.md` and `Template - JUTI-OTH.docx`. It appears to predate the transition to the IMRaD/Typst format.

No paper-compilation command is documented because this checkout does not include the required Typst executable or the `juti` template dependencies. Install these external build dependencies to compile `paper/main.typ`. The executable analysis pipeline does not depend on the paper build environment.

---

## Notes: Generated and Non-Generated Artifacts

- Experiment runners generate scenario-specific split manifests at runtime; no standalone split-generation CLI is provided. Existing manifests in `data/splits/` record the current assignments generated with `seed=42`.
- This guide covers preprocessing, embedding extraction, model fitting, calibration/evaluation, LoRA ablation, data audits, and aggregate SVG generation (Sections 2–7).
- Dataset downloads require network access and compliance with the source dataset terms. Model downloads require access to the pinned Hugging Face revision.
- Repeated bootstrap/resampling runs are deterministic for the recorded seed, provided that dependencies versions, device behavior, and input files remain unchanged.

## 2026-10-30 - CRC-family comparison integrated

- Wired CRC, W-CRC, and IW-LTT procedures into the primary/secondary study runner. An algebra check confirmed that unit calibration and test weights make W-CRC identical to CRC, so only the CRC result is presented; each head records certificate status, threshold, and holdout sensitivity/specificity under `risk_control_comparison`.
- The RAD-DINO primary TBX11K CRC holdout result is 90.9%/70.2% (logistic) and 91.5%/81.5% (MLP); IW-LTT gave 94.5%/51.8% and 93.3%/68.0%. On Shenzhen, CRC gave 86.3%/68.1% and 85.7%/76.1%; IW-LTT gave 97.0%/31.3% and 94.0%/51.5%.
- Fixed the IW-LTT candidate sequence using 25 unique source-validation score quantiles, chosen independently of the target calibration scores; the chosen grid is stored in the run JSON. This avoids selecting the fixed-sequence hypotheses from the same calibration scores being tested.
- Used unit density ratios and upper bound one as the explicit exchangeable baseline. This does not adjust for site shift; valid positive-class density ratios and an upper bound are still required for that claim.
- Updated reproduction, pipeline, research-process, and paper documentation. Generated RAD-DINO baseline comparisons for the primary TBX11K and secondary Shenzhen protocols in separate run directories; the new JSON contains the baseline fields.

## 2026-10-31 — Identity-aware robustness and preprocessing audits

- Corrected patient-unit language and removed the separate unit-weight W-CRC row because it is algebraically identical to CRC. The current analysis remains image-level; the identity audit does not establish distinct patients.
- Added `scripts/run_robustness_audits.py` to repeat source heads across ten seeds, split all target positives into fresh calibration/evaluation pools, evaluate the expanded Tier A/CRC/IW-LTT size grid, report Clopper–Pearson Monte Carlo intervals and the Tier A binomial floor, estimate positive-class domain odds only as diagnostics, run a within-site random control, and simulate the WSR null type-I rate. Run with `python -m scripts.run_robustness_audits` as shown in `REPRODUCE.md`.
- Added `scripts/probe_pixel_site_features.py` as a native-image site-origin comparator and image-level pixel alias exclusion. The current five-fold probe reached 100% balanced accuracy on 8,934 images; native size/aspect and simple grayscale summaries separate these three datasets, but this does not establish causal model shortcut use.
- Added preprocessing options for percentile-windowing all grayscale images and aspect-preserving letterbox resize. `REPRODUCE.md` records the cache and embedding commands; only results with matching cache provenance should be compared.
- Completed all four alias-excluded preprocessing-ablation origin probes. ResNet50-A1 window-all-224/letterbox-224 and RAD-DINO window-all-518/letterbox-518 achieved balanced accuracies of 98.4%, 98.7%, 98.3%, and 99.2%, respectively; none of these transformations removed readily decodable site information. The probes measure site information in embeddings and do not establish causal shortcutting. JSON outputs are in `experiments/runs/preprocessing_ablation/`.
- Added a preliminary STARD-AI reporting gap checklist and a data-governance record. Institutional ethics status and permission to distribute image-linked derived outputs remain unresolved and should block public artifact release.

## 2026-10-31 — Label sensitivity and verification-budget workflows

- Added scripts/run_ambiguous_label_sensitivity.py to embed the 140 official TBX11K obsolete/unresolved records and compare extreme mappings that treat every ambiguous finding as active TB versus non-active TB. The run remains image-level, excludes exact-pixel duplicate aliases from target evaluation, and reports no PAC claim.
- Added scripts/run_selective_budget_sweep.py to resample independent calibration and evaluation score pools from the de-duplicated TBX11K training split across screening budgets 1,000–6,496, prevalences 0.05–0.40, and low-score verification probabilities 0.10–1.00. This is an empirical bootstrap simulation, not validation on a new patient cohort.
- Added an exact 8-bit histogram percentile implementation for the uniform-windowing ablation; it produces the same linear percentile values as NumPy without sorting the full grayscale array.
- Completed both 1,000-repetition label-sensitivity runs. Each examines 140 obsolete/unresolved TBX11K images under all-positive and all-negative mappings, excludes 25 exact-pixel aliases from validation evaluation, and explicitly makes no biological-label or PAC claim; the encoder-specific results are in `experiments/runs/label_sensitivity/`.
- Completed the corrected RAD-DINO budget sweep with independent empirical-bootstrap calibration/evaluation pools, four screening budgets, four prevalences, four low-score verification probabilities, and 100 repetitions per cell. The naive arm's finite-pool violation rate ranged from 2% to 100%; the four-bound arm refused some cells and had 0–2% observed violations among issued certificates. These simulation frequencies are not future-population guarantees.
- Completed 10-seed source-head and 1,000-repetition nested target-pool audits for both encoders. The target-positive pools are randomly split into disjoint calibration/evaluation halves (660 TBX11K positives; 336 Shenzhen positives); exact-pixel aliases are excluded from evaluation. All results remain image-level because patient linkage is unverified. IW-LTT used only 20 repetitions and 10 orderings per cell, so that comparison remains preliminary.
- Across the 2,000-repetition WSR null simulation, 50 tests rejected (2.5%; CP95 1.86–3.28%). Cross-fitted diagnostic effective sample sizes were 415/660 for TBX11K and 43/336 for Shenzhen under ResNet50-A1, and 447/660 and 69/336 under RAD-DINO; these odds are not validated density ratios and were not supplied to weighted procedures.
- Completed the independent seed-43 LoRA run with the same 3-epoch configuration and 1,000 Tier A repetitions. Its source-only B0 was 95.1%/89.9%; at $n^+=200$ its mean sensitivity/specificity was 91.1%/94.5% and the finite-pool violation frequency was 17.7% (177/1,000; CP95 15.4–20.2%), versus 1.2% for seed 42. The large difference is an image-level audit result, not validation or disproof of patient-population PAC coverage; see `p1_2_lora_seed43/results.json`.
- REPRODUCE.md records the commands, completed outputs, and interpretation limits for these analyses.
