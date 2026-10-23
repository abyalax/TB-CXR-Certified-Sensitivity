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
