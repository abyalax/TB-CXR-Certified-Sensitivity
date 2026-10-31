# Literature-search

PubMed and IEEE Xplore queries below were recorded as exploratory searches on
21 October 2026. The search is designed to challenge, not establish, the
narrow novelty question: finite-sample class-conditional sensitivity
calibration for site-level TB-CXR under known-probability selective
verification.

| Index | Database/interface | Exact search string                                                                                                                                                                | Disposition                                                                                             |
| ----- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| C1    | PubMed             | `("tuberculosis" OR "TB") AND ("chest x-ray" OR "chest radiograph" OR CXR) AND ("artificial intelligence" OR "computer-aided detection") AND (threshold OR calibration OR triage)` | Retained multi-site accuracy and local-threshold studies.                                               |
| C2    | PubMed             | `("tuberculosis" OR "TB") AND ("chest x-ray" OR "chest radiograph") AND ("verification bias" OR "selective verification" OR "reference standard bias")`                            | No retrieved item was identified as the same finite-sample PAC procedure. This is not proof of absence. |
| C3    | IEEE Xplore        | `("tuberculosis" OR TB) AND ("chest x-ray" OR "chest radiograph") AND (threshold OR calibration OR "domain shift")`                                                                | Screening-method papers found; institutional full-text screening remains required.                      |
| C4    | Scopus (ON HOLD)   | `TITLE-ABS-KEY((tuberculosis OR TB) AND ("chest x-ray" OR "chest radiograph" OR CXR) AND (threshold OR calibration OR "risk control" OR conformal OR "verification bias"))`        | Not executed. Resume only when requested and institutional access is available.                         |

## Scopus work to perform when resumed

Run each search separately in Scopus Advanced Search, record the exact date,
interface, query, result count, and export RIS/CSV with DOI, title, abstract,
year, and author fields. Do not merge counts until deduplication is complete.

1. `TITLE-ABS-KEY((tuberculosis OR TB) AND ("chest x-ray" OR "chest radiograph" OR CXR) AND ("artificial intelligence" OR "computer-aided detection" OR CAD) AND (threshold OR calibration OR triage OR sensitivity))`
2. `TITLE-ABS-KEY((tuberculosis OR TB) AND ("chest x-ray" OR "chest radiograph" OR CXR) AND ("verification bias" OR "partial verification" OR "selective verification" OR "imperfect reference standard"))`
3. `TITLE-ABS-KEY(("class-conditional" OR "conditional risk" OR "sensitivity guarantee" OR "high-probability risk control" OR PAC OR "learn-then-test") AND (calibration OR threshold OR "risk control") AND (medical OR clinical OR imaging OR radiology))`
4. `TITLE-ABS-KEY(("conformal risk control" OR "weighted conformal" OR "importance weighting") AND ("covariate shift" OR "distribution shift") AND (sensitivity OR recall OR "false negative"))`

After export: deduplicate by DOI, then normalized title; screen title/abstract
against population (TB/CXR or transferable statistical method), threshold or
risk-control method, and verification/shift assumptions; retrieve full text
for included candidates; log exclusion reasons and any novelty-relevant
counterexample. Save the export and screening table locally without including
restricted full-text documents in this repository. Until completed, make no
systematic-review or priority claim.

Representative records retained for the working draft include multi-site
accuracy evaluation (Qin et al.) and local-threshold analysis in an
individual-participant meta-analysis (Tavaziva et al.). These do not establish
novelty for the present procedure. The manuscript must use cautious language
such as “our exploratory search did not identify,” never assert priority.

## Supplemental discovery search — 31 October 2026

These searches used the web index for discovery and citation verification; result counts are not database exports and must not be described as comprehensive screening. Google Scholar was not directly searchable through the available interface (the direct Scholar URL was inaccessible), and Scopus remains on hold. No systematic-review, exhaustive-search, or priority claim is supported.

| Interface | Query used | Relevant retrieval or disposition |
| --- | --- | --- |
| PubMed via web index | `PubMed TB chest x-ray threshold 95% sensitivity local threshold external validation calibration multi site` | Retrieved multi-site local-threshold studies and a 2025 universal-versus-stratified CAD threshold evaluation; these motivate local assessment but do not establish finite-sample PAC validity. |
| IEEE Xplore via web index | `site:ieeexplore.ieee.org tuberculosis chest x-ray sensitivity calibration threshold multi site` | Retrieved general TB-CXR model papers, but no directly matching finite-sample class-conditional certify-or-refuse method was identified in this limited discovery search. |
| Google Scholar via web index | `site:scholar.google.com/scholar TBX11K sensitivity threshold` | No direct Scholar result was accessible; this query is not a completed Google Scholar search. |
| Garuda via web index | `site:garuda.kemdikbud.go.id tuberculosis chest x-ray sensitivity` | Retrieved general imaging/classification and clinical radiography records; none identified as the same PAC threshold method. This is not exhaustive Garuda screening. |
| arXiv via web index | `site:arxiv.org tuberculosis chest x-ray sensitivity threshold target 95% calibration risk control` | Retrieved Akhtar et al. (arXiv:2609.21763), a relevant 2026 counterexample: source-selected thresholds met the 95% sensitivity point estimate in only 4 of 16 target evaluations. It addresses model-score portability rather than the present finite-sample positive-class calibration procedure. |

The 2026 arXiv record narrows the motivation: external threshold portability has already been audited on TBX11K, so this project must describe its contribution as a procedural implementation and exploratory audit of finite-sample certify-or-refuse calibration and selective verification, not as the first evidence that thresholds may fail to transport. Full-text screening, exact native database queries, exports, and Scopus remain outstanding.

## Selected reference-metadata check — 31 October 2026

Confirmed selected title/author/year/venue/DOI or arXiv metadata against primary records: [Qin et al. in PubMed](https://pubmed.ncbi.nlm.nih.gov/31628424/), [Tavaziva et al. in PubMed](https://pubmed.ncbi.nlm.nih.gov/34286831/), [Akhtar et al. on arXiv](https://arxiv.org/abs/2609.21763), [Almeida et al. in PMLR](https://proceedings.mlr.press/v266/almeida25a.html), [RAD-DINO in Nature Machine Intelligence](https://www.nature.com/articles/s42256-024-00965-w), and [Clopper–Pearson in Oxford Academic](https://academic.oup.com/biomet/article-abstract/26/4/404/291538). The TBX11K Montgomery citation details were checked against the [NLM dataset readme](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/NLM-MontgomeryCXRSet-ReadMe.pdf). Corrected Tavaziva's first author and page range plus the Qin title and Bates page range in `paper/references.bib`. This is a targeted metadata check, not complete citation-context or literature screening; Wilks and the protected State of the Art §9.4 entry still require independent review before submission.

## Follow-up reference audit — 31 October 2026

Checked the TBX11K and dataset-method citations against the [official CVF CVPR 2020 record](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Rethinking_Computer-Aided_Tuberculosis_Diagnosis_CVPR_2020_paper.html), [IEEE Xplore CVPR record](https://ieeexplore.ieee.org/abstract/document/9156613), [PubMed record for the TBX11K journal paper](https://pubmed.ncbi.nlm.nih.gov/37934644/), and [NLM PubMed records for Candemir et al.](https://pubmed.ncbi.nlm.nih.gov/24239990/) and [Jaeger et al.](https://pubmed.ncbi.nlm.nih.gov/24108713/). Added the CVPR DOI, corrected the TBX11K journal year to its 2024 issue year in the duplicate draft bibliography, and aligned the Candemir and Jaeger author lists with their indexed records. Confirmed the [IMS Learn Then Test article](https://www.imstat.org/publications/aoas/aoas_19_2/aoas_19_2.pdf), the [Project Euclid Wilks record](https://doi.org/10.1214/aoms/1177731788), and the [arXiv metadata for Akhtar et al.](https://arxiv.org/abs/2609.21763). This corrects selected bibliographic fields and checks the context of the principal claims cited in the active draft; full reference-context review, the protected State of the Art §9.4 entry, and Scopus/Google Scholar searches remain open.
