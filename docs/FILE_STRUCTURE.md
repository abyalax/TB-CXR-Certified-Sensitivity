```
└── 📁tb-cxr-certified-sensitivity
    └── 📁artifacts
        └── 📁audits
            ├── rad_dino_training_overlap_110cbc18.json
        └── 📁checkpoints
            └── 📁rad-dino-110cbc18
                └── 📁.cache
                    └── 📁huggingface
                        └── 📁download
                            ├── config.json.metadata
                            ├── model.safetensors.lock
                            ├── preprocessor_config.json.metadata
                            ├── xGOKKLRSlIhH692hSVvI1-gpoa8=.dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae.incomplete
                        ├── .gitignore
                ├── checkpoint_metadata.json
                ├── config.json
                ├── model.safetensors
                ├── preprocessor_config.json
        └── 📁embeddings
            └── 📁chest_xray-110cbc18d513
                ├── embeddings.npy
                ├── manifest.csv
                ├── metadata.json
            └── 📁general-767268603ca0
                ├── embeddings.npy
                ├── manifest.csv
                ├── metadata.json
    └── 📁configs
        ├── data.yaml
        ├── method.yaml
        ├── model.yaml
    └── 📁data
        └── 📁processed
            └── 📁primary
                └── 📁images
                    └── 📁montgomery
                        ├── 0145b07f1ea1eacf3770aede.png
                        ├── ..........png
                    └── 📁shenzhen
                        ├── 0046f593b8a8df00c58c7485.png
                        ├── ..........png
                    └── 📁tbx11k
                        ├── 00061b6ff3aed33a3f792db0.png
                        ├── ..........png
                ├── hash_audit.json
                ├── manifest.csv
                ├── preprocessing.json
            └── 📁rad_dino_518
                └── 📁images
                    └── 📁montgomery
                        ├── 0145b07f1ea1eacf3770aede.png
                        ├── ...........png
                    └── 📁shenzhen
                        ├── 0046f593b8a8df00c58c7485.png
                        ├── ............png
                    └── 📁tbx11k
                        ├── 00061b6ff3aed33a3f792db0.png
                        ├── .............png
                ├── hash_audit.json
                ├── manifest.csv
                ├── preprocessing.json
        └── 📁raw
            └── 📁montgomery
                └── 📁ClinicalReadings
                    ├── MCUCXR_0001_0.txt
                    ├── ..............txt
                    ├── MCUCXR_0399_1.txt
                └── 📁CXR_png
                    ├── MCUCXR_0001_0.png
                    ├── ..............png
                    ├── MCUCXR_0399_1.png
                └── 📁ManualMask
                    └── 📁leftMask
                        ├── MCUCXR_0001_0.png
                        ├── ..............png
                        ├── MCUCXR_0399_1.png
                ├── montgomery_consensus_roi.csv
                ├── NLM-MontgomeryCXRSet-ReadMe.pdf
            └── 📁shenzhen
                └── 📁Annotations
                    └── 📁Annotations_json
                        └── 📁SeparateFiles
                            ├── CHNCXR_0327_1.json
                            ├── ..............json
                            ├── CHNCXR_0662_1.json
                        ├── Annotations_AllinOne_json.json
                    └── 📁masks
                        ├── CHNCXR_0327_1_Calcified_Nodule_2.png
                        ├── ......png
                        ├── CHNCXR_0662_1_Small_Infiltrate_(non-linear)_1.png
                    ├── Readme.docx
                    ├── Statistics_ShenzhenDataset.csv
                └── 📁Annotations-2
                    └── 📁Annotations_json
                        └── 📁SeparateFiles
                            ├── CHNCXR_0327_1.json
                            ├── .....json
                            ├── CHNCXR_0662_1.json
                        ├── Annotations_AllinOne_json_US.json
                    └── 📁masks
                        ├── CHNCXR_0327_1_Small_Infiltrate_(non-linear)_1.png
                        ├── ........png
                        ├── CHNCXR_0662_1_Small_Infiltrate_(non-linear)_1.png
                    ├── Readme.docx
                    ├── Statistics_ShenzhenDataset_US.csv
                └── 📁ClinicalReadings
                    ├── CHNCXR_0001_0.txt
                    ├── ..............txt
                    ├── CHNCXR_0662_1.txt
                └── 📁CXR_png
                    ├── CHNCXR_0001_0.png
                    ├── ..............png
                    ├── CHNCXR_0662_1.png
                ├── NLM-ChinaCXRSet-ReadMe.docx
                ├── shenzhen_consensus_roi.csv
            ├── TBX11K.zip
        └── 📁splits
            ├── lora_source_seed42.csv
            ├── primary_tbx11k_loso_image_split_seed42.csv
            ├── primary_tbx11k_source_image_split_seed42.csv
            ├── random_image_control_seed42.csv
            ├── secondary_shenzhen_loso_image_split_seed42.csv
        ├── README.md
    └── 📁experiments
        └── 📁runs
            └── 📁primary_tbx11k_loso
                └── 📁chest_xray-110cbc18d513
                    └── 📁p0_8_p1_linear_selective
                        ├── results.json
                    └── 📁p0_8_selective
                        ├── results.json
                    └── 📁p1_2_lora
                        └── 📁adapter
                            ├── adapter_config.json
                            ├── adapter_model.safetensors
                            ├── README.md
                        ├── results.json
                    └── 📁p1_2_lora_final
                        └── 📁adapter
                            ├── adapter_config.json
                            ├── adapter_model.safetensors
                            ├── README.md
                        ├── classifier_head.pt
                        ├── results.json
                    ├── p0_4_origin_probe.json
                    ├── results.json
                └── 📁general-767268603ca0
                    ├── results.json
            └── 📁random_image_control
                └── 📁chest_xray-110cbc18d513
                    ├── results.json
            └── 📁secondary_shenzhen_loso
                └── 📁chest_xray-110cbc18d513
                    └── 📁p0_9_lesion_size
                        ├── results.json
                    ├── results.json
                └── 📁general-767268603ca0
                    ├── results.json
    └── 📁figures
    └── 📁notebooks
    └── 📁paper
        └── 📁.tools
            └── 📁fonts
                ├── liberation_serif.zip
            └── 📁typst
                └── 📁typst-x86_64-pc-windows-msvc
                    ├── LICENSE
                    ├── NOTICE
                    ├── README.md
                    ├── typst.exe
            ├── page-1.png
            ├── typst-v0.15.1.zip
        └── 📁figures
            ├── latest_results.svg
            ├── study_workflow.svg
        ├── build.ps1
        ├── imrad.md
        ├── literature_search.md
        ├── main.pdf
        ├── main.typ
        ├── README.md
        ├── references.bib
        ├── setup.typ
        ├── template.typ
    └── 📁SA-CTC for Tuberculosis
        └── 📁Literature Review
            ├── A Joint Finite-Sample Certificate for Adaptive Selective Conformal Risk Control.pdf
            ├── A multi-country head-to-head accuracy comparison of automated chest X-ray algorithms for tuberculosis.pdf
            ├── Accuracy of computer-aided chest X-ray in community-based tuberculosis screening - Lessons from the 2016 Kenya National Tuberculosis Prevalence Survey.pdf
            ├── Active, anytime-valid risk controlling prediction sets.pdf
            ├── Audit-First VAPO - Risk-Certified Selective Updates under Imperfect Verification.pdf
            ├── Beyond Benchmark Scores - Auditing Medical Vision-Language Models for Chest X-Ray Tuberculosis Screening.pdf
            ├── Beyond overall accuracy - subgroup performance and threshold calibration of Al-assisted chest radiography for equity-oriented pulmonary tuberculosis screening.pdf
            ├── Certify or Refuse - A Cross-Model Map for Selective Risk Control with Coverage Floors under Covariate Shift.pdf
            ├── Chest X-ray Analysis With Deep Learning-Based Software as a Triage Test for Pulmonary Tuberculosis - An Individual Patient Data Meta-Analysis of Diagnostic Accuracy.pdf
            ├── Computer-aided detection of tuberculosis from chest.pdf
            ├── Computer-aided detection pragmatic threshold calibration in TB active case finding.pdf
            ├── Conformal Prediction Beyond Exchangeability.pdf
            ├── Conformal Prediction Under Covariate Shift.pdf
            ├── Conformal Risk Control.pdf
            ├── Deep learning for automated classification of tuberculosis-related chest - dataset distribution shift limits diagnostic performance generalizability.pdf
            ├── DG-UCTB - A Domain-Generalized Uncertainty-Calibrated Framework for Tuberculosis Detection in Chest Radiographs.pdf
            ├── False Sense of Safety in Selective Signal Classification - Auditing Bound Tightness and Exchangeability for Risk Control.pdf
            ├── Few-Shot Cross-Dataset Adaptation for Tuberculosis Detection Using DenseNet.pdf
            ├── Generalization and Informativeness of Weighted Conformal Risk Control Under Covariate Shift.pdf
            ├── High Probability Risk Control Under Covariate Shift.pdf
            ├── Limitations of Public Chest - Label Quality, Domain Shift, Bias and Evaluation Challenges.pdf
            ├── Literature Review.md
            ├── Model-agnostic Selective Labeling with Provable Statistical Guarantees.pdf
            ├── PAC - Prediction Sets Under Covariate Shift.pdf
            ├── PAC - Prediction Sets Under Label Shift.pdf
            ├── Performance of universal and stratified computer-aided.pdf
            ├── Probabilistic Conformal Coverage Guarantees in Small-Data Settings.pdf
            ├── Reliable Algorithm Selection for Machine Learning-Guided Design.pdf
            ├── Selective Conformal Risk Control.pdf
            ├── Sharp training-conditional coverage for conformal prediction under covariate shift.pdf
            ├── Simultaneous alleviation of verification and reference standard biases in a community-based tuberculosis screening study using Bayesian latent class analysis.pdf
            ├── Source-free domain adaptation for multi-institutional chest.pdf
            ├── State Of The Art.md
            ├── Weighted Bayesian Conformal Prediction.pdf
            ├── When Average Calibration Fails - Site-Conditional Federated Conformal Risk Control.pdf
        ├── Certify-or-Refuse Sensitivity Calibration for TB Chest Radiographs.pdf
        ├── IMRAD.md
        ├── main.pdf
        ├── SA-CTC for Tuberculosis.md
        ├── Template - JUTI-OTH.docx
    └── 📁scripts
        └── 📁__pycache__
            ├── analyze_lesion_size.cpython-312.pyc
            ├── probe_dataset_origin.cpython-312.pyc
            ├── run_primary_real_study.cpython-312.pyc
        ├── analyze_lesion_size.py
        ├── audit_rad_dino_training_overlap.py
        ├── probe_dataset_origin.py
        ├── render_result_thumbnails.py
        ├── run_lora_ablation.py
        ├── run_primary_real_study.py
        ├── run_random_split_control.py
    └── 📁src
        └── 📁tb_certify
            └── 📁__pycache__
                ├── __init__.cpython-312.pyc
                ├── __main__.cpython-312.pyc
                ├── cli.cpython-312.pyc
            └── 📁calibration
                └── 📁__pycache__
                    ├── __init__.cpython-312.pyc
                    ├── risk_control.cpython-312.pyc
                    ├── source_threshold.cpython-312.pyc
                    ├── tier_a.cpython-312.pyc
                ├── __init__.py
                ├── risk_control.py
                ├── source_threshold.py
                ├── tier_a.py
            └── 📁data
                └── 📁__pycache__
                    ├── __init__.cpython-312.pyc
                    ├── nlm.cpython-312.pyc
                    ├── pipeline.cpython-312.pyc
                    ├── preprocessing.cpython-312.pyc
                    ├── records.cpython-312.pyc
                    ├── roi.cpython-312.pyc
                    ├── splitting.cpython-312.pyc
                    ├── tbx11k.cpython-312.pyc
                ├── __init__.py
                ├── nlm.py
                ├── pipeline.py
                ├── preprocessing.py
                ├── records.py
                ├── roi.py
                ├── splitting.py
                ├── tbx11k.py
            └── 📁evaluation
                └── 📁__pycache__
                    ├── __init__.cpython-312.pyc
                    ├── binomial.cpython-312.pyc
                    ├── metrics.cpython-312.pyc
                    ├── selective_study.cpython-312.pyc
                    ├── tier_a_study.cpython-312.pyc
                ├── __init__.py
                ├── binomial.py
                ├── metrics.py
                ├── selective_study.py
                ├── tier_a_study.py
            └── 📁models
                └── 📁__pycache__
                    ├── __init__.cpython-312.pyc
                    ├── embeddings.cpython-312.pyc
                    ├── provenance.cpython-312.pyc
                ├── __init__.py
                ├── embeddings.py
                ├── provenance.py
            └── 📁training
                └── 📁__pycache__
                    ├── __init__.cpython-312.pyc
                    ├── source_head.cpython-312.pyc
                ├── __init__.py
                ├── source_head.py
            └── 📁verification
                └── 📁__pycache__
                    ├── __init__.cpython-312.pyc
                    ├── selective.cpython-312.pyc
                ├── __init__.py
                ├── selective.py
            ├── __init__.py
            ├── __main__.py
            ├── cli.py
    └── 📁tests
        └── 📁__pycache__
            ├── test_origin_probe.cpython-312-pytest-9.1.1.pyc
            ├── test_risk_control.cpython-312-pytest-9.1.1.pyc
            ├── test_selective_study.cpython-312-pytest-9.1.1.pyc
            ├── test_selective.cpython-312-pytest-9.1.1.pyc
        ├── test_origin_probe.py
        ├── test_risk_control.py
        ├── test_selective_study.py
        ├── test_selective.py
    ├── .gitignore
    ├── AGENTS.md
    ├── CONTEXT.md
    ├── download_dataset.ps1
    ├── LICENSE
    ├── pyproject.toml
    ├── README.md
    ├── TODO.md
    └── uv.lock
```
