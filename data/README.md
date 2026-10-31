# Dataset Source

| Dataset    | Berkas utama                  | Keterangan                                                                                   |
| ---------- | ----------------------------- | -------------------------------------------------------------------------------------------- |
| Montgomery | `data/raw/montgomery/`        | 138 CXR, ClinicalReadings, lung masks, dan ROI                                               |
| Shenzhen   | `data/raw/shenzhen/`          | 662 CXR, ClinicalReadings, anotasi, dan ROI                                                  |
| TBX11K     | `data/raw/TBX11K.zip` (~3 GB) | 8.400 citra pada split resmi train+validation; 8.260 dipakai setelah filter label penelitian |

Run Download Dataset From Source with 

```PowerShell
./download_dataset.ps1
```

### [Shenzen Hospital CXR and Montgomery County CXR](https://www.lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayDatasets.html)

## [Montgomery County CXR Set](https://f10409.github.io/RadHarmony/datasets/montgomery_cxr.html)

_138 radiograf: 80 normal dan 58 abnormal dengan manifestasi yang konsisten dengan TB menurut dokumentasi NLM. Label ini bersifat radiologis, bukan hasil konfirmasi mikrobiologis._

> Format citra: PNG. Label 0/1 pada nama file menunjukkan normal/abnormal, bukan pengganti pemeriksaan label klinis.

## [Shenzen Hospital CXR Set](https://f10409.github.io/RadHarmony/datasets/shenzhen_cxr.html)

_662 radiograf: 326 normal dan 336 abnormal dengan manifestasi yang konsisten dengan TB menurut dokumentasi NLM. Label ini bersifat radiologis, bukan hasil konfirmasi mikrobiologis._

> Format citra: PNG. Gunakan clinical readings untuk membentuk label evaluasi.

### [TBX11K](https://github.com/yun-liu/Tuberculosis)

_Split resmi train+validation berisi tepat 8.400 citra: 6.600 train + 1.800 validation. Daftar `TBX11K_trainval.txt` terdiri dari 3.800 `health` + 3.800 `sick` + 800 `tb`. README upstream menyebut total dataset 11.200 citra dan 2.800 citra test (6.600 train + 1.800 validation + 2.800 test)._

> Pipeline penelitian memakai `TBX11K_trainval.txt` saja, bukan daftar `all_*` yang mencampur situs eksternal. Dari 8.400 citra train+validation, 140 citra di folder `tb` dikeluarkan karena anotasinya hanya obsolete/unresolved atau tidak memenuhi aturan positif `ActiveTuberculosis`; hasil akhirnya **8.260 citra**: 660 `active_tb` + 3.800 `healthy` + 3.800 `non_tb_disease`. Citra test tidak digunakan karena ground truth test tidak dirilis. Arsip lokal juga memuat 583 citra `imgs/extra` dari dataset lain dan berkas metadata/anotasi; hitungan file arsip atau semua gambar di folder bukan jumlah sampel penelitian.

# Interpretation

## Montgomery County (MC) Dataset

- Characteristics: This dataset was collected from a tuberculosis screening program in Montgomery County, Maryland, USA.
- Data Structure: Consists of 138 frontal CXR images (80 normal and 58 cases with TB manifestations).
- Representation: The public collection includes image files and radiology readings. The project has not verified a patient crosswalk or a one-image-per-patient mapping from those files.
- References :
  - [https://pmc.ncbi.nlm.nih.gov/articles/PMC4256233/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4256233/)
  - [https://www.mdpi.com/2075-4418/16/13/1947](https://www.mdpi.com/2075-4418/16/13/1947)
  - [https://pmc.ncbi.nlm.nih.gov/articles/PMC13065592/](https://pmc.ncbi.nlm.nih.gov/articles/PMC13065592/)

## Shenzhen Hospital Dataset

- Characteristics: Collected from Shenzhen No. 3 People's Hospital in China.
- Data Structure: Consists of 662 frontal CXR images (326 normal and 336 TB cases).
- Representation: The public collection includes image files and clinical readings. The project has not verified a patient crosswalk or a one-image-per-patient mapping from those files.&#x20;
- References :
  - [https://pmc.ncbi.nlm.nih.gov/articles/PMC9645800/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9645800/)
  - [https://www.mdpi.com/2075-4418/16/13/1947](https://www.mdpi.com/2075-4418/16/13/1947)
  - [https://pmc.ncbi.nlm.nih.gov/articles/PMC13065592/](https://pmc.ncbi.nlm.nih.gov/articles/PMC13065592/)

## TBX11K Dataset

- Characteristics: A large-scale dataset developed by Nankai University to address data quantity limitations in deep learning models for TB detection.&#x20;
    [https://www.researchgate.net/publication/343466325_Rethinking_Computer-Aided_Tuberculosis_Diagnosis](https://www.researchgate.net/publication/343466325_Rethinking_Computer-Aided_Tuberculosis_Diagnosis)
- Data Structure: Consists of 11,200 CXR images divided into several categories (healthy, non-TB disease, active TB, latent TB).
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC13065592/](https://pmc.ncbi.nlm.nih.gov/articles/PMC13065592/)
- Representation: The project assets do not include a verified patient crosswalk for TBX11K. Dataset-level statements about unique individuals do not identify those individuals in the local image manifests.
    [https://mlmed.org/torchxrayvision/datasets.html](https://mlmed.org/torchxrayvision/datasets.html)
    [https://www.sid.ir/fileserver/je/1-344462-x-1516120.pdf](https://www.sid.ir/fileserver/je/1-344462-x-1516120.pdf)

# Folder Structure

```
└── 📁data
    └── 📁processed
        └── 📁primary
            └── 📁images
                └── 📁montgomery
                    ├── 0145b07f1ea1eacf3770aede.png
                    ├── 020dedd89d4ea5a36f14532e.png
                    ├── .......png
                └── 📁shenzhen
                    ├── 0046f593b8a8df00c58c7485.png
                    ├── ff2dcf24d1601739bcc1cfc5.png
                    ├── .......png
                └── 📁tbx11k
                    ├── 00061b6ff3aed33a3f792db0.png
                    ├── 000717a7c463445228ac5724.png
                    ├── .......png
            ├── hash_audit.json
            ├── manifest.csv
            ├── preprocessing.json
        └── 📁rad_dino_518
            └── 📁images
                └── 📁montgomery
                    ├── 0145b07f1ea1eacf3770aede.png
                    ├── 020dedd89d4ea5a36f14532e.png
                    ├── ......png
                └── 📁shenzhen
                    ├── 0046f593b8a8df00c58c7485.png
                    ├── 0089f1fea7f3907e3ca068f1.png
                    ├── ........png
                └── 📁tbx11k
                    ├── 00061b6ff3aed33a3f792db0.png
                    ├── 000717a7c463445228ac5724.png
                    ├── ........png
            ├── hash_audit.json
            ├── manifest.csv
            ├── preprocessing.json
    └── 📁raw
        └── 📁montgomery
            └── 📁ClinicalReadings
                ├── MCUCXR_0001_0.txt
                ├── MCUCXR_0002_0.txt
                ├── ..............txt
                ├── MCUCXR_0399_1.txt
            └── 📁CXR_png
                ├── MCUCXR_0001_0.png
                ├── MCUCXR_0002_0.png
                ├── ..............png
                ├── MCUCXR_0399_1.png
            └── 📁ManualMask
                └── 📁leftMask
                    ├── MCUCXR_0001_0.png
                    ├── MCUCXR_0002_0.png
                    ├── ..............png
                    ├── MCUCXR_0399_1.png
                └── 📁rightMask
                    ├── MCUCXR_0001_0.png
                    ├── MCUCXR_0002_0.png
                    ├── ..............png
                    ├── MCUCXR_0399_1.png
            ├── montgomery_consensus_roi.csv
            ├── NLM-MontgomeryCXRSet-ReadMe.pdf
        └── 📁shenzhen
            └── 📁Annotations
                └── 📁Annotations_json
                    └── 📁SeparateFiles
                        ├── CHNCXR_0327_1.json
                        ├── CHNCXR_0328_1.json
                        ├── ..............json
                        ├── CHNCXR_0662_1.json
                    ├── Annotations_AllinOne_json.json
                └── 📁masks
                    ├── CHNCXR_0327_1_Calcified_Nodule_2.png
                    ├── CHNCXR_0327_1_Clustered_Nodule_(2mm-5mm_apart)_1.png
                    ├── .................................png
                    ├── CHNCXR_0662_1_Small_Infiltrate_(non-linear)_1.png
                ├── Readme.docx
                ├── Statistics_ShenzhenDataset.csv
            └── 📁Annotations-2
                └── 📁Annotations_json
                    └── 📁SeparateFiles
                        ├── CHNCXR_0327_1.json
                        ├── CHNCXR_0328_1.json
                        ├── ..............json
                        ├── CHNCXR_0662_1.json
                    ├── Annotations_AllinOne_json_US.json
                └── 📁masks
                    ├── CHNCXR_0327_1_Small_Infiltrate_(non-linear)_1.png
                    ├── CHNCXR_0328_1_Cavity_1.png
                    ├── CHNCXR_0662_1_Linear_Density_2.png
                    ├── CHNCXR_0662_1_Small_Infiltrate_(non-linear)_1.png
                ├── Readme.docx
                ├── Statistics_ShenzhenDataset_US.csv
            └── 📁ClinicalReadings
                ├── CHNCXR_0001_0.txt
                ├── CHNCXR_0002_0.txt
                ├── ..............txt
                ├── CHNCXR_0662_1.txt
            └── 📁CXR_png
                ├── CHNCXR_0001_0.png
                ├── CHNCXR_0002_0.png
                ├── ..............png
                ├── CHNCXR_0662_1.png
            ├── NLM-ChinaCXRSet-ReadMe.docx
            ├── shenzhen_consensus_roi.csv
        ├── TBX11K.zip
    └── 📁splits
        ├── primary_tbx11k_loso_image_split_seed42.csv
        ├── primary_tbx11k_source_image_split_seed42.csv
        └── secondary_shenzhen_loso_image_split_seed42.csv
```

# License and Citation Dataset

## Mountgomery County CXR

1. Candemir S, Jaeger S, Musco J, Xue Z, Karargyris A, Antani SK, Thoma GR, Palaniappan K. Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. IEEE Trans Med Imaging. 2014 doi: 10.1109/TMl.2013.2290491. PMID: 24239990
2. Jaeger S, Karargyris A, Candemir S, Folio L, Siegelman J, Callaghan FM, Xue Z, Palaniappan K, Singh RK, Antani SK. Automatic tuberculosis screening using chest radiographs. IEEE Trans Med Imaging. 2014 doi: 10.1109/TMl.2013.2284099. PMID: 24108713

## Shenzen Hospital CXR

1. Jaeger S, Karargyris A, Candemir S, Folio L, Siegelman J, Callaghan F, Xue Z, Palaniappan K, Singh RK, Antani S, Thoma G, Wang YX, Lu PX, McDonald CJ.  Automatic tuberculosis screening using chest radiographs. IEEE Trans Med Imaging. 2014 Feb;33(2):233-45. doi: 10.1109/TMI.2013.2284099. PMID: 24108713
2. Candemir S, Jaeger S, Palaniappan K, Musco JP, Singh RK, Xue Z, Karargyris A, Antani S, Thoma G, McDonald CJ. Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. IEEE Trans Med Imaging. 2014 Feb;33(2):577-90. doi: 10.1109/TMI.2013.2290491. PMID: 24239990

## TBX11K

```bib
@article{liu2023revisiting,
  title={Revisiting Computer-Aided Tuberculosis Diagnosis},
  author={Liu, Yun and Wu, Yu-Huan and Zhang, Shi-Chen and Liu, Li and Wu, Min and Cheng, Ming-Ming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={46},
  number={4},
  pages={2316--2332},
  year={2023},
  publisher={IEEE}
}

@inproceedings{liu2020rethinking,
  title={Rethinking Computer-aided Tuberculosis Diagnosis},
  author={Liu, Yun and Wu, Yu-Huan and Ban, Yunfeng and Wang, Huifang and Cheng, Ming-Ming},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2646--2655},
  year={2020}
}
```
