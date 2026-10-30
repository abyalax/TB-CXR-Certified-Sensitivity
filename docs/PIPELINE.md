Workflow Research

**1. Collect Datasets** (`data/raw/`)
- Montgomery, Shenzhen (dengan ClinicalReadings, CXR_png, ManualMask/Annotations ganda, file ROI konsensus), dan TBX11K.zip
- `download_dataset.ps1` script for download data from source data.

**2. Preprocessing & split** (`data/processed/`, `data/splits/`)
- Dua versi processed: `primary` (ukuran asli) dan `rad_dino_518` (khusus resolusi 518 untuk RAD-DINO) â€�? masing-masing dengan `manifest.csv`, `hash_audit.json`, `preprocessing.json` untuk traceability
- Setiap image merepresentasikan satu pasien (lihat `data/README.md`), sehingga split berbasis image juga menjaga pemisahan pasien. Split di-generate per skenario eval: LOSO (leave-one-source-out) untuk primary (TBX11K) dan secondary (Shenzhen), source split, random control, dan split khusus sumber LoRA â€�? semua dengan `seed42` (reproducible)

**3. Feature extraction / embedding** (`artifacts/embeddings/`)
- Dua encoder pralatih yang dibekukan (frozen): `chest_xray-110cbc18d513` (RAD-DINO, 518x518, CLS embedding 768 dimensi) dan `general-767268603ca0` (ResNet50-A1 ImageNet, pembanding)
- Encoder utama tidak di-fine-tune pada data TB dalam jalur eksperimen utama; checkpoint RAD-DINO disimpan dan diverifikasi hash-nya sebelum ekstraksi embedding

**4. Audit data leak** (`artifacts/audits/`)
- `audit_rad_dino_training_overlap.py` cek apakah data eval overlap dengan data training RAD-DINO (penting karena RAD-DINO model pretrained, bukan dilatih dari nol)
- Probe asal-site dari embedding dijalankan oleh `scripts/probe_dataset_origin.py:28-85`: mengambil embedding frozen dan label `site` dari manifest (`:30-42`), lalu memakai `StandardScaler` + balanced Logistic Regression (`:43-56`) dengan Stratified 5-fold cross-validation (`:57-60`). Ini mengukur informasi domain yang dapat dibaca dari embedding, bukan prediksi TB atau bukti penyebab shortcut. Hasil RAD-DINO tercatat di `experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_4_origin_probe.json`: accuracy 0.9926 (≈99.3%), balanced accuracy 0.9800, macro OVR AUC 0.9992. Angka 99.3% adalah akurasi cross-validated probe untuk asal-site.

**5. Training head, calibration, dan verification**
- **Encoder fitur:** RAD-DINO dan ResNet50-A1 ImageNet dibekukan pada studi utama; ekstraksi ada di `src/tb_certify/models/embeddings.py`. Fine-tuning RAD-DINO LoRA adalah ablation di `scripts/run_lora_ablation.py`.
- **Training head:** `src/tb_certify/training/source_head.py` melatih standardized Logistic Regression dari embedding data site sumber; MLP menjadi pembanding. Site target yang diuji dikecualikan dari training head. Ini classifier di atas embedding, bukan encoder.
- **Threshold baseline:** dipilih dari validasi source di `src/tb_certify/training/source_head.py` (`choose_source_threshold`); bersifat empiris, bukan sertifikat.
- **Kalibrasi ambang:** `src/tb_certify/calibration/source_threshold.py` menangani threshold empiris; `src/tb_certify/calibration/tier_a.py` mengimplementasikan Tier A; `src/tb_certify/calibration/risk_control.py` mengimplementasikan marginal CRC, W-CRC, dan IW-LTT. Runner membandingkan ketiganya berbobot satu dan mengukur performa pada holdout terpisah. Kisi IW-LTT ditetapkan dari kuantil skor validasi sumber, independen dari kalibrasi target. Bobot satu adalah baseline exchangeable, bukan koreksi pergeseran situs; shift nyata memerlukan rasio kepadatan positif dan batas bobot tervalidasi. Sertifikasi ini statistik (Clopper-Pearson/order statistic), bukan model machine learning.
- **Selective verification (certify-or-refuse):** `src/tb_certify/verification/selective.py` mengimplementasikan four-CP dan linear importance-weighted Hoeffding UCB. Bound IPW $V \cdot Y \cdot (\mathbb{1}\{\text{score} < t\} - \alpha)/\pi(\text{score})$ dihitung per threshold; threshold lolos berurutan jika UCB ≤ 0, jika tidak sertifikat menolak.
- **Evaluasi:** `src/tb_certify/evaluation/selective_study.py` menjalankan selective verification dan mengevaluasi threshold pada data terpisah; `src/tb_certify/evaluation/tier_a_study.py` menjalankan studi Tier A; `src/tb_certify/evaluation/binomial.py` menghitung interval Clopper-Pearson dan `src/tb_certify/evaluation/metrics.py` metrik threshold.
- **Provenance:** `src/tb_certify/models/provenance.py` menyimpan/memvalidasi asal model dan data; sidecar provenance embedding ditulis dari `src/tb_certify/models/embeddings.py`.

**6. Eksperimen** (`experiments/runs/`)
- **Alur umum run klasifikasi:** muat embedding frozen (RAD-DINO atau ResNet50-A1 sesuai run) dan manifest split; fit head Logistic Regression atau MLP pada data train; pilih threshold baseline dari validasi sumber untuk target sensitivitas 0.90; hitung metrik pada validasi dan evaluasi target. Pada LOSO, site target tidak masuk training head. Setiap eksperimen mencatat seed, ukuran cohort/split, metadata encoder, threshold, dan hasil per head di `results.json`.
- **`primary_tbx11k_loso`:** leave-one-site-out dengan TBX11K sebagai target dan site lain sebagai sumber. Dari target, protokol memisahkan data kalibrasi positif dan evaluasi; menjalankan baseline threshold empiris serta studi Tier A pada berbagai ukuran kalibrasi/repetisi. Untuk TBX11K juga ada negative controls (`healthy_only`, `healthy_plus_non_tb`). Selective verification eksploratori hanya dihitung jika `--selective-repetitions` diaktifkan (nilai default 0).
- **`secondary_shenzhen_loso`:** protokol leave-one-site-out serupa dengan Shenzhen sebagai target; head dilatih pada site sumber, lalu threshold baseline dan Tier A dievaluasi pada target. Hasil per head mencakup hitungan cohort, metrik baseline, dan keluaran Tier A.
- **`random_image_control`:** kontrol dengan pembagian acak terstratifikasi tingkat pasien dan site tercampur. Logistic Regression dan MLP dievaluasi pada split evaluasi; threshold berasal dari validasi, dan Tier A memakai split kalibrasi tersendiri. Tujuannya membandingkan dengan LOSO, bukan menguji transportabilitas lintas site.
- **Metrik threshold / confusion matrix:** untuk setiap threshold evaluasi, `src/tb_certify/evaluation/metrics.py` menghitung TP, FN, TN, FP, sensitivity, dan specificity. Komponen confusion matrix ini diserialisasi pada `results.json` sebagai `source_validation_metrics`, `target_b0_metrics` (atau `evaluation_b0_metrics` di kontrol random), dan metrik negative control jika berlaku. Pipeline belum membuat gambar atau tabel confusion matrix terpisah, dan metrik ini dihitung pada threshold yang ditentukan protokol.
- **Ablasi LoRA (`p1_2_lora`, `p1_2_lora_final`):** fine-tune RAD-DINO dengan LoRA sebagai pembanding terhadap encoder frozen, kemudian evaluasi classifier head pada target yang ditentukan. Hasil dan konfigurasi ablasinya disimpan pada folder run masing-masing.
- **Selective verification (`p0_8_selective`, `p0_8_p1_linear_selective`):** membandingkan Tier A naif, four-CP verification-bias certificate, dan linear IPW-Hoeffding certificate. Untuk linear IPW-Hoeffding ada run khusus di `experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_8_p1_linear_selective/`. Keluaran utamanya adalah threshold yang lolos/ditolak, bound sertifikasi, dan perilaku refuse; ini bukan confusion matrix.
- **Analisis ukuran lesi (`p0_9_lesion_size`):** analisis sensitivitas threshold pada strata ukuran lesi subset gambar positif beranotasi ROI; ini analisis subgroup, bukan training model klasifikasi baru.
- Tiap kondisi run menyimpan hasil pada `results.json` di direktori run tersendiri. Artefak tambahan seperti split manifest atau metadata model dapat berada di sampingnya; ringkasan visual hasil, bila dibuat, merupakan langkah terpisah.

**7. Scripts orkestrasi** (`scripts/`)
- `scripts/run_primary_real_study.py` — menjalankan studi leave-site-out: melatih head dari site sumber, mengkalibrasi threshold, lalu mengevaluasi sertifikasi pada site target.
- `scripts/run_random_split_control.py` — menjalankan kontrol dengan split gambar acak yang mencampur site untuk membandingkan hasil saat distribution shift tidak dijaga.
- `scripts/run_lora_ablation.py` — fine-tune RAD-DINO dengan LoRA pada data sumber dan mengaudit performa head hasilnya pada TBX11K.
- `scripts/analyze_lesion_size.py` — menganalisis sensitivitas threshold menurut strata ukuran lesi pada subset gambar positif yang memiliki anotasi ROI.
- `scripts/probe_dataset_origin.py` — menguji seberapa akurat asal-site dapat diprediksi dari embedding frozen menggunakan cross-validation.

**8. Penulisan paper** (`paper/`)
- `imrad.md` sebagai draf aktif di-compile ke `main.typ` pakai Typst (ada binary `typst.exe` sendiri di `.tools/`)
- `literature_search.md`, `references.bib` untuk sitasi
- Folder `SA-CTC for Tuberculosis/` kelihatan seperti draf lama/paralel (ada `Literature Review.md`, `Template - JUTI-OTH.docx`) â€�? mungkin draf versi awal sebelum pivot ke format IMRaD Typst

**9. Testing** (`tests/`)
- Unit test untuk tiap modul kalibrasi/verifikasi (`test_risk_control.py`, `test_selective.py`, dll.) â€�? jaga validitas metode sebelum dipakai di eksperimen real

**raw data preprocess+split (dua resolusi) ekstrak embedding (dua model) audit overlap jalankan studi kalibrasi di beberapa skenario (LOSO, random control, LoRA ablation) simpan hasil per run tulis ke IMRaD/Typst**.
