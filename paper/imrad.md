# Certify-or-Refuse Sensitivity Calibration for TB Chest Radiographs

**Working research manuscript (IMRAD).** This is the active manuscript file.
The JUTI Typst draft is not updated by this workflow. All real-data estimates
below are exploratory image-level results; patient linkage has not been verified. Clinical validity is not claimed.

## Abstract

**Background:** A chest-radiograph tuberculosis (TB) triage threshold chosen
at a source site may not retain sensitivity at a new site, and selective microbiological
confirmation can distort the observed positive calibration sample.

**Methods:** We implemented the exact target-positive order-statistic
certificate (Tier A) for target sensitivity at least 90% with failure
probability 5%; it refuses below a 29-positive floor. Frozen RAD-DINO and ResNet50-A1 image
features with source-only logistic and multilayer perceptron (MLP) heads were
evaluated on official TBX11K train and validation pools. We also audited a
site-mixed image-random negative control, simulated selective verification
with Clopper--Pearson (CP) bounds and an inverse-probability-weighted (IPW)
Hoeffding comparator, and multi-seed and nested-resampling audits.

**Results:** At the source threshold, TBX11K validation sensitivity and specificity were 91.5%/64.3% for logistic regression and 95.7%/57.5% for the MLP with RAD-DINO. As comparator, ResNet50-A1 reached 84.1%/33.3% and 88.4%/31.0%, respectively. At $n^+=100$ in Shenzhen, RAD-DINO MLP Tier A averaged 92.1%/62.9% sensitivity/specificity. The MLP violation rate and the probability of specificity at least 70% were both 4.9%, compared with a 28.8% logistic violation rate. On Shenzhen, source-threshold sensitivity was below 90% for both heads (79.2% logistic; 77.4% MLP). A source-only RAD-DINO low-rank adaptation (LoRA) ablation at seed 42 reached 95.7%/88.8% at its source threshold. At $n^+=200$, its Tier A mean was 93.8%/94.0%, with 1.2% violation (12/1,000; CP95 0.6–2.1%). Independent seed 43 reached B0 95.1%/89.9%, but its $n^+=200$ finite-pool violation rate was 17.7% (177/1,000; CP95 15.4–20.2%). This difference is an image-level audit result and neither validates nor disproves population PAC coverage. The random-split deduplicated site-mixed control had 99.1%/98.9% specificity at the source threshold (logistic/MLP). Under simulated low-score verification probability $q=0.10$, naive calibration violated the empirical sensitivity target in 100% of 100 audit repetitions; the four-bound certificate issued in 4%/7% of repetitions with no observed violations among issued thresholds. The linear IPW--Hoeffding arm always refused. At $q=1$, the pre-specified specificity-gap utility gate failed (11.8 and 22.6 percentage-point gaps for logistic regression and MLP).


**Conclusion:** The implementation makes the exact sample-size refusal rule
explicit and demonstrates the utility cost of conservative selective-label
certification. All conclusions remain exploratory image-level analyses;
patient linkage is unverified. External database screening, ethics review, and data-use review
remain outstanding before submission or artifact release.

## 1. Introduction

Computer-aided detection (CAD) scores from chest radiographs (CXR) are used to
support TB screening and triage. A decision threshold that reaches a desired
sensitivity in a development cohort need not preserve that sensitivity in a
new site with different acquisition, population, or negative-disease spectrum.
In field screening, microbiological verification may also depend on symptoms,
score, or clinician judgment. If lower-scoring people are less likely to be
verified, the observed TB-positive sample is selected rather than representative.

Prior multi-site CXR evaluation and an individual-participant diagnostic-accuracy meta-analysis have examined performance and threshold behavior across populations ([Qin et al., 2019](https://pubmed.ncbi.nlm.nih.gov/31628424/); [Tavaziva et al., 2022](https://pubmed.ncbi.nlm.nih.gov/34286831/)). A 2026 exploratory audit also reports that a TBX11K-selected 95% sensitivity threshold met that point estimate in only 4 of 16 target evaluations ([Akhtar et al., 2026](https://arxiv.org/abs/2609.21763)). Our work studies a different question: finite-sample certify-or-refuse calibration of positive-class sensitivity and selective verification. Literature screening is incomplete, and we make no priority claim (see [`literature_search.md`](literature_search.md)).

Risk-control comparators motivate finite-sample calibration and covariate-shift corrections ([Bates et al., 2021](https://doi.org/10.1145/3478535); [Zecchin et al., 2025](https://arxiv.org/abs/2501.11413); [Almeida et al., 2025](https://proceedings.mlr.press/v266/almeida25a.html); [Angelopoulos et al., 2025](https://doi.org/10.1214/24-AOAS1998)). The runner compares marginal conformal risk control (CRC) and importance-weighted learn-then-test (IW-LTT) using unit density weights and evaluates issued thresholds on an independent target holdout. Unit-weight W-CRC is algebraically identical to CRC and is not reported as a separate result. These are exchangeable positive-class baselines, not shift-adjusted comparisons. A real-site shift analysis still requires a justified positive-conditional density ratio and valid upper bound, which are unavailable. RAD-DINO is a self-supervised chest-X-ray encoder ([Pérez-García et al., 2025](https://doi.org/10.1038/s42256-024-00965-w)).

This study investigates a certify-or-refuse workflow: calibrate a threshold
using target-site TB-positive scores and issue it only if an exact
finite-sample class-conditional sensitivity criterion is feasible. For selective
verification, we compare a conservative four-bound certificate and a linear
importance-weighted Hoeffding arm under a simulated known-probability policy.
We do not introduce a new theorem. The contribution is an implementation and
audit of the assumptions, refusal behavior, and empirical utility trade-offs.

Research questions are:

1. What sensitivity/specificity trade-off does the target-only exact rule show
   on an image-level target holdout?
2. How different is a site-mixed random image split from leave-site-out
   evaluation?
3. How do naive and conservative procedures behave when low-score reference
   verification is selective with known probability?

## 2. Methods

### 2.1 Data sources, label policy, and units

The processed cohort contains 9,060 images: 138 Montgomery, 662 Shenzhen,
and 8,260 binary TBX11K records. Montgomery and Shenzhen positives are abnormal radiographs with TB-consistent manifestations under the documented clinical-reading and filename-suffix rule; these are radiological labels, not microbiological confirmation. The official TBX11K train-validation lists
contain 8,400 images; 139 obsolete-only and one unresolved TB image were
excluded, leaving 8,260 records. Montgomery and Shenzhen image-level labels
follow their documented filename suffix convention. For TBX11K, active TB is
positive only when at least one `ActiveTuberculosis` annotation is present; 139 obsolete-only and one unresolved image were excluded.
The official training and validation lists are used; the unlabeled test
partition and mixed-site `all_*` lists are excluded.

The primary protocol trains on Montgomery and Shenzhen, calibrates on
TBX11K-train positives, and evaluates on TBX11K validation. The secondary
protocol holds Shenzhen out, trains on Montgomery and official TBX11K train,
and uses an image-stratified 50/50 Shenzhen calibration/evaluation split; each
half contains 168 positive images (and 163 negative images).
For model inputs, we retained the full chest radiograph and applied no
lung-field or lesion-ROI crop. Images were converted to grayscale; 8-bit
intensities were retained, while higher-bit images were mapped to uint8 using
a per-image 0.5th–99.5th percentile window. The grayscale image was then
resized to 224 × 224 with Lanczos interpolation and replicated to three
channels. This cache resolution was used by ResNet50; RAD-DINO inputs were
processed at 518 × 518.
Montgomery/Shenzhen region-of-interest (ROI) CSVs annotate positive lesions only; they are not
complete label inventories. Patient linkage has not been verified from source records; `patient_id` remains blank. The recorded sampling unit is one image. Patient linkage is unverified and the explicit `patient_id` column remains blank.

### 2.2 Encoders and source-only decision rules

We used frozen ImageNet ResNet50-A1 and RAD-DINO encoders with pinned
revisions. RAD-DINO used the 518-pixel processor and CLS-token embedding;
embeddings were extracted on CUDA with 16-bit floating-point (FP16) inference
and stored as float32.
Logistic regression and a one-hidden-layer MLP (128 units) were fitted only
to source training images. A source validation set selected the empirical
threshold whose observed source-positive sensitivity reached 90%. This
source threshold is an empirical baseline, not a target-site certificate; we
call the source-threshold baseline B0. For the primary run, the source head
was trained on 639 images (Montgomery and Shenzhen) and selected its threshold
on 161 source-validation images. This small source cohort limits the precision
of the site-shift interpretation and precludes a broad claim about a general
site effect.

An origin probe used five-fold stratified cross-validation to predict site
from RAD-DINO embeddings. It diagnoses whether embeddings contain site
information; it cannot establish causal shortcut learning.

### 2.3 CRC-family baseline comparison

For each fitted head, the primary and secondary runners calibrate marginal CRC
and IW-LTT on the full target-positive calibration pool, then evaluate
issued thresholds on the independent target holdout. The comparison uses unit
density ratios and upper bound one, representing the exchangeable positive-class
baseline. CRC uses target FNR tolerance $alpha=0.10$; IW-LTT also uses
failure probability $delta=0.05$. W-CRC uses unit test weights, so deployment
points share the same certificate. These results do not adjust for covariate
shift. W-CRC at unit calibration and test weights is algebraically identical
to CRC, so it is not presented as a separate result. A valid shift-adjusted analysis requires a justified target/source
density ratio conditional on TB positivity and a valid bound; neither is
available here. The runner records certificate status, threshold, diagnostics,
and holdout sensitivity/specificity under `risk_control_comparison` in each
head's `results.json`. For IW-LTT, its 25 candidate thresholds are the unique
0.01--0.99 quantiles of source-validation scores, fixed independently of the
target calibration sample. The WSR p-value averages 100 seeded random
orderings (seed 42 in the reported run).

### 2.4 Exact Tier A certificate

Let $s(x)$ increase with evidence for active TB, and predict positive when
$s(x)\ge t$. For a target population $P_T$, the sensitivity estimand is

$$
\operatorname{Sens}_T(t)
= P_T\{s(X)\ge t\mid Y=1\}, \qquad
\operatorname{FNR}_T(t)=1-\operatorname{Sens}_T(t).
$$

The target false-negative-rate (FNR) tolerance is $\alpha=0.10$ and the
failure probability is $\delta=0.05$. Given $n^+$ target-positive calibration scores
$S_1^+,\ldots,S_{n^+}^+$, define

$$
k^*(n^+)=\max\left\{k\in\{0,\ldots,n^+-1\}:
\Pr\bigl[\operatorname{Bin}(n^+,\alpha)\le k\bigr]\le\delta\right\}.
$$

If this set is empty, output **REFUSE**. Otherwise, with ascending order
statistics $S_{(1)}^+\le\cdots\le S_{(n^+)}^+$, set

$$
\widehat t=S_{(k^*(n^+)+1)}^+.
$$

Under representative i.i.d. target-positive calibration observations, a
fixed score/model, accurate reference labels, and stationarity between
calibration and use, the finite-sample statement is

$$
\Pr_{D_{\mathrm{cal}}}\!\left[
\operatorname{Sens}_T(\widehat t)\ge 1-\alpha
\right]\ge 1-\delta.
$$

For $k=0$, feasibility requires $(1-\alpha)^{n^+}\le\delta$, hence

$$
n^+_{\min}=\left\lceil\frac{\log\delta}{\log(1-\alpha)}\right\rceil=29.
$$

This order-statistic tolerance-limit calculation follows the distribution-free finite-sample framework of [Wilks (1941)](https://doi.org/10.1214/aoms/1177731788). At a tied boundary score, the `>=` decision includes all tied cases, making the realized calibration miss count no larger than the allowed order-statistic count.

The guarantee concerns population sensitivity conditional on the assumptions;
it is not a per-patient guarantee. This is a probably approximately correct
(PAC) statement. Because verified patient linkage is absent, our real-data
runs audit the algorithm and its holdout behavior but do not validate this
PAC interpretation for patients.

We refer to this exact target-positive order-statistic procedure as Tier A.

### 2.5 Selective-verification policy and certificates

For each fully labeled record in the TBX11K-train simulation, a fixed
reference score cutoff $c$ was used. All records with $s_i\ge c$ were
verified, while records below $c$ were independently verified with
probability $q$. Thus the known verification probability is

$$
\pi_i=\Pr(V_i=1\mid s_i)=
\begin{cases}1,&s_i\ge c,\\q,&s_i<c.\end{cases}
$$

The experiment hides unverified labels from the calibrators. The threshold
reference cutoff $c$ is the source-validation threshold. The candidate grid
contains unique source-validation score quantiles at 25 probability points
from 0.01 through 0.99, fixed before simulated verification labels are
examined and tested in increasing score-threshold order (from more sensitive
to less sensitive). The single failure probability $\delta=0.05$ applies to
the complete pre-specified fixed sequence. The risk is nondecreasing on this grid, and the rule issues only its passing prefix. If an issued threshold were unsafe, the first unsafe grid point would also have passed its pointwise four-component bound; this monotone step-up argument controls false issuance without dividing $\delta$ by the number of candidates. The audited low-score verification probabilities are
$q\in\{0.10,0.25,0.50,1.00\}$.

We refer to the four-bound verification-bias certificate as VB. The exact binomial interval construction follows [Clopper and Pearson (1934)](https://doi.org/10.1093/biomet/26.4.404). It splits
$\delta$ into four tails
$\eta=\delta/4$. At candidate threshold $t$, bound high-score and low-score
miss counts above with CP upper bounds $U_H(t)$ and $U_L(t)$; scale the
low-score upper bound by $1/q$. Bound high- and low-score positive totals
below with CP lower bounds $L_H,L_L$, scaling the low-score term by $1/q$.
The implementation's FNR upper bound is

$$
U_{\mathrm{FNR}}(t)=
\frac{U_H(t)+U_L(t)/q}{L_H+L_L/q}.
$$

If the denominator lower bound is zero, the bound is infinite. A threshold
passes when $U_{\mathrm{FNR}}(t)\le\alpha$; a fixed-sequence rule issues the
largest threshold in the initial consecutive passing prefix, otherwise it
refuses. For misses below $c$, the implementation uses the lower segment
$(s_i<\min(t,c))$; all records at or above $c$ must be observed.

The linear arm tests a single IPW functional per fixed threshold. With
$Z_i(t)=V_iY_i(\mathbf1\{s_i<t\}-\alpha)/\pi_i$, it computes

$$
\overline Z(t)+
\sqrt{\frac{\log(1/\delta)\sum_{i=1}^N\pi_i^{-2}}{2N^2}},
$$

and passes if this upper bound is at most zero. The implementation stops at the first failing threshold. This is the evaluated Hoeffding arm, not a claim that it is uniformly tighter or empirically useful. When low-score verification probability $q$ is small, the confidence bonus includes terms proportional to $q^{-2}$; at finite screening counts this can exceed the negative empirical margin, explaining the observed refusal as a conservative negative result rather than a failed validity check.

### 2.6 Controls, secondary analysis, and uncertainty summaries

N1 denotes healthy-only negatives, and N2 denotes healthy plus non-TB disease
negatives in the negative-spectrum control. The random-split negative control uses a
stratified image-level random split (patient linkage unverified) that allows each site to occur in model
training, calibration, validation, and evaluation; it is intentionally not
a transportability estimate or a patient-level validation.

For lesion size, we used the 250 boxes in the project’s Shenzhen ROI CSV and grouped them by image across the 68 matched active-TB images. The public Shenzhen archive contains separate annotation collections for two radiologists, each covering 336 TB-positive images; available documentation does not establish how the project CSV’s boxes were derived from those collections. When an image had multiple boxes, we used the largest single annotated box (not the geometric union of boxes); size is its area divided by the original full-image area. Images are split into three approximately equal tertiles. This proxy is not clinical severity and annotations do not cover all positives. We report exact 95% CP binomial intervals for the number detected within each stratum.

Tier A studies repeatedly sample distinct positives without replacement from
the finite calibration pool, with 1,000 repetitions for the main and random
split evaluations. Selective-verification policies have 100 repetitions.
Empirical violation frequency is computed against the finite target holdout;
its CP interval describes Monte Carlo frequency uncertainty, not uncertainty
from sampling an infinite patient population.

### 2.7 LoRA ablation protocol

The RAD-DINO LoRA arm is an optional source-only fine-tuning ablation. It targets the query and value attention projections with rank 8, alpha 16, dropout 0.05, FP16 CUDA autocast, and a source-trained classification head. Training uses Montgomery/Shenzhen source-train images only; source validation sets the baseline threshold. Official TBX11K train positives are reserved for Tier A calibration and TBX11K validation for evaluation. Both runs use three epochs, AdamW (learning rate $10^{-4}$, weight decay 0.01), batch size 1 with gradient accumulation 4, and independent seeds 42 and 43. Each uses 639 source-train images and 161 source-validation images; 295,681 parameters are trainable (adapter plus classifier head). Each run directory contains the adapter, classifier head, and full JSON result; source split manifests are stored separately under `data/splits/`.

### 2.8 Multi-seed and nested target-pool audits

For each frozen encoder, source heads and thresholds were refit at seeds 42–51. For each target, all observed positive images were repeatedly divided into disjoint calibration and evaluation halves (660 TBX11K positives or 336 Shenzhen positives before splitting), and 1,000 repetitions evaluated Tier A and CRC over $n^+\in\{10,20,28,29,\ldots,45,46,61,76,100\}$. Exact-pixel aliases were excluded from target negative evaluation. IW-LTT used 20 repetitions and 10 random orderings per cell because of its runtime; those results are preliminary. Cross-fitted domain-classifier odds and ESS are diagnostic summaries only and are not treated as validated density ratios or supplied to a weighted guarantee.

### 2.9 Ambiguous-label and verification-budget sensitivity

The label-sensitivity audit remapped all 140 TBX11K obsolete/unresolved records (139 obsolete and one unresolved; 104 official train and 36 validation) to active TB in one extreme scenario and non-active TB in the other. It excluded 25 exact-pixel aliases from validation evaluation and reran the source threshold and Tier A at $n^+\in\{29,46,61,100,200\}$ for both encoders. The verification-budget sweep independently bootstrapped calibration and evaluation scores from the deduplicated TBX11K training split across $N\in\{1000,2000,4000,6496\}$, prevalence $\{0.05,0.10,0.20,0.40\}$, and low-score verification probability $q\in\{0.10,0.25,0.50,1.00\}$, with 100 repetitions per cell. Both analyses are image-level empirical stress tests, not validation of patient independence or future-population PAC guarantees.

## 3. Results

### 3.1 Cohort and pipeline

The 9,060-image cohort passed cache integrity checks; the 518-pixel RAD-DINO
cache decoded all images and matched recorded SHA-256 values. Audit of
882,775 upstream RAD-DINO training-list records found no image-ID, stem, or
basename match to local manifests. This metadata-only audit cannot rule out
renamed pixel duplicates.

### 3.2 Source-threshold and target-only results

| Protocol / target                               |        Head | Sensitivity | Specificity | Notes                                                           |
| ----------------------------------------------- | ----------: | ----------: | ----------: | --------------------------------------------------------------- |
| TBX11K official validation, RAD-DINO B0         |    Logistic |       91.5% |       64.3% | 164 positive / 1,600 negative images                            |
| TBX11K official validation, RAD-DINO B0         |         MLP |       95.7% |       57.5% | Source threshold                                                |
| TBX11K official validation, ResNet50-A1 B0      |    Logistic |       84.1% |       33.3% | Source threshold                                                |
| TBX11K official validation, ResNet50-A1 B0      |         MLP |       88.4% |       31.0% | Source threshold                                                |
| Shenzhen image holdout, RAD-DINO B0             |    Logistic |       79.2% |       84.0% | Source threshold misses 90% target; 168 positive / 163 negative |
| Shenzhen image holdout, RAD-DINO B0             |         MLP |       77.4% |       89.6% | Source threshold misses 90% target; 168 positive / 163 negative |
| Deduplicated site-mixed random split, RAD-DINO B0 | Logistic | 90.0% | 99.1% | 211 positive / 1,576 negative images |
| Deduplicated site-mixed random split, RAD-DINO B0 | MLP | 91.5% | 98.9% | Not a transportability control |
| TBX11K validation, source-only RAD-DINO LoRA B0 | Linear head |       95.7% |       88.8% | 164 positive / 1,600 negative images                            |

In the primary TBX11K Tier A analysis, at $n^+=29$ mean sensitivity was
96.0% (logistic) and 97.0% (MLP); mean specificity was 41.8% and 39.3%.
At $n^+=200$, means were 93.2%/58.8% and 94.5%/65.7%, respectively.
Across $n^+=29,46,61,100,200$, the estimated probability that certified
specificity reached 70% was 6.1%, 4.0%, 5.1%, 2.7%, and 2.3% for logistic
regression, and 11.7%, 15.7%, 19.4%, 14.9%, and 33.2% for the MLP.
In the Shenzhen secondary analysis, the source threshold missed the 90%
sensitivity target: sensitivity/specificity were 79.2%/84.0% for logistic
regression and 77.4%/89.6% for MLP. Given the 168 calibration positives
reported in Methods, $n^+=200$ deterministically refuses. At $n^+=100$, the Shenzhen MLP
mean was 92.1% sensitivity/62.9% specificity with 4.9% empirical violation;
the logistic mean was 93.8%/39.9% with 28.8% violation. The probability of
specificity reaching 70% at $n^+=100$ was also 4.9% for MLP and 0.6% for
logistic. The equal 4.9% values are separate resampling summaries: one is
violation frequency and the other is the frequency of specificity at least
70%.

At Shenzhen $n^+=29,46,61,100$, these probabilities were 10.4%, 10.3%,
10.3%, and 4.9% for MLP, and 4.0%, 1.6%, 0.9%, and 0.6% for logistic.
All are estimates from 1,000 image-level resamples (patient linkage unverified); exact values, violation
intervals, and refusal frequencies are in the linked run JSON files below.

The alias-excluded site-origin probe reached 99.3% accuracy, 98.6% balanced accuracy, and
99.9% macro one-vs-rest area under the receiver-operating-characteristic
(ROC) curve over RAD-DINO embeddings. This indicates site information is present but does not identify its source or causal mechanism. The deduplicated random-split source-threshold metrics were heterogeneous by site:

The four preprocessing-ablation origin probes excluded 126 exact-pixel aliases and retained high site decodability: balanced accuracy was 98.4% for ResNet50-A1 window-all-224, 98.7% for ResNet50-A1 letterbox-224, 98.3% for RAD-DINO window-all-518, and 99.2% for RAD-DINO letterbox-518 ([results](../experiments/runs/preprocessing_ablation/)). These probes measure site information in embeddings and do not identify its cause or prove causal TB-label shortcutting.

| Evaluation site | Positive / negative images | Logistic sensitivity / specificity | MLP sensitivity / specificity |
| --- | ---: | ---: | ---: |
| Montgomery | 15 / 11 | 53.3% / 100.0% | 60.0% / 100.0% |
| Shenzhen | 74 / 58 | 86.5% / 93.1% | 89.2% / 89.7% |
| TBX11K | 122 / 1,507 | 96.7% / 99.3% | 96.7% / 99.3% |

The split excludes 126 exact-pixel duplicate aliases, all TBX11K non-TB images. Site-specific rates describe this image split; they are not estimates for deployment populations.
N1 contains 800 healthy negatives; N2 contains 800 healthy and 800 non-TB disease negatives. At the primary logistic source threshold, specificity was 85.6% against
N1 (healthy-only negatives) compared with 64.3% against N2 (healthy plus
non-TB disease); for the MLP it was 76.8% versus 57.5%. Sensitivity was
unchanged within each head because the positive holdout was identical. This
confirms substantial dependence of specificity on negative spectrum; the
N2 spectrum is a constructed dataset mixture and does not represent a deployment population.

![Latest aggregate result preview generated from experiment JSON](figures/latest_results.svg)

### 3.3 Selective-verification results

The naive results illustrate the selection effect induced by this simulated policy; they are not an independent empirical validation. With 100 policy repetitions, naive Tier A violation was 100% for logistic and MLP at $q=0.10$. At $q=0.25$, naive violation was 100% and 77%,
respectively. The four-bound VB method had zero observed violations among
issued runs; at $q=0.10$ its issue rates were 4% (logistic) and 7% (MLP),
with specificity means 23.1% and 10.5% in those few issued runs. At
$q=0.25$, issue rate was 100% for both heads, with zero observed violations.
The linear IPW--Hoeffding arm refused every threshold in all audited policy
runs. At $q=0.10$, zero violations among four issued VB certificates has a two-sided exact 95% CP interval with an upper endpoint near 60.2%; this is uninformative about the nominal 5% population failure bound.

At $q=1$, a pre-specified tightening utility gate required VB mean
specificity to be no more than 3 percentage points below naive Tier A. It failed: logistic
VB specificity was 51.8% versus naive 63.6% (gap 11.8 points); MLP VB was
51.1% versus 73.7% (gap 22.6 points). The VB arm is therefore conservative
in the audits but the tightening/utility gate is not passed.

### 3.4 CRC-family comparison

On the RAD-DINO TBX11K primary holdout, CRC selected a threshold with logistic
regression sensitivity/specificity of 90.9%/70.2% and MLP sensitivity/specificity
of 91.5%/81.5%. Unit-weight W-CRC is algebraically identical to CRC and is not
reported as a separate result. Unit-weight IW-LTT selected a lower
threshold: logistic regression had 94.5% sensitivity and 51.8% specificity,
and the MLP had 93.3% sensitivity and 68.0% specificity. The IW-LTT p-values
at the selected thresholds were 0.00000577 and 0.00918, respectively, at nominal
$delta=0.05$. On the Shenzhen holdout, CRC had sensitivity/specificity
of 86.3%/68.1% for logistic regression and 85.7%/76.1% for the MLP. IW-LTT
had 97.0%/31.3% and 94.0%/51.5%, with p-values 0.00784 and 0.000113.
These are empirical holdout results under unit weights, not evidence of
correction for site shift. The comparison calibrates on target positives, so it
does not replace a source-to-target weighted transport analysis.

### Supplementary exploratory lesion-size ROI analysis

Parentheses give exact 95% Clopper--Pearson confidence intervals.

| Head     |             Small tertile |            Medium tertile |             Large tertile |
| -------- | ------------------------: | ------------------------: | ------------------------: |
| Logistic | 20/23 = 87.0% (66.4-97.2) | 20/23 = 87.0% (66.4-97.2) | 19/22 = 86.4% (65.1-97.1) |
| MLP      | 22/23 = 95.7% (78.1-99.9) | 21/23 = 91.3% (72.0-98.9) | 20/22 = 90.9% (70.8-98.9) |

Intervals are wide and overlap. Sensitivity was flat for logistic and decreased for MLP across annotated
area; the small positive-only subset
cannot establish absence of a size effect or generalize to unannotated
positive images.

![Study workflow and assumption boundary](figures/study_workflow.svg)

### 3.6 LoRA ablation

The RTX 3050 6 GB run completed without an out-of-memory error. Training
binary cross-entropy decreased from 0.6013 to 0.2472 to 0.1222 over three fixed epochs.
The empirical source threshold was 0.9352, with source-validation sensitivity 91.3% and specificity
98.8%. On TBX11K validation it gave 95.7% sensitivity (157/164) and 88.8%
specificity (1,420/1,600). With Tier A target-positive calibration,
sensitivity/specificity means were 96.3%/73.9% at $n^+=29$, 95.4%/83.1% at
46, 95.0%/87.5% at 61, 94.9%/89.7% at 100, and 93.8%/94.0% at 200.
The empirical violation frequency and Monte Carlo 95% CP interval at each
calibration size were:

| $n^+$ | Mean sensitivity | Mean specificity | Violation rate (count/1,000; 95% CP interval) |
| ----: | ---------------: | ---------------: | --------------------------------------------: |
|    29 |            96.3% |            73.9% |                     4.6% (46/1,000; 3.4–6.1%) |
|    46 |            95.4% |            83.1% |                     5.6% (56/1,000; 4.3–7.2%) |
|    61 |            95.0% |            87.5% |                     3.8% (38/1,000; 2.7–5.2%) |
|   100 |            94.9% |            89.7% |                     1.5% (15/1,000; 0.8–2.5%) |
|   200 |            93.8% |            94.0% |                     1.2% (12/1,000; 0.6–2.1%) |

At $n^+=46$, the finite-holdout violation estimate is above 5%; these audit
frequencies neither prove nor disprove the assumed population PAC statement.
Peak allocated CUDA memory was 502,656,512 bytes.

The earlier `p1_2_lora` and retained `p1_2_lora_final` folders are same-seed reruns at seed 42. Both produced the same TBX11K B0 counts and rounded sensitivity/specificity (95.7%/88.8%); the final threshold differs by 0.00012. Independent seed 43 produced similar B0 sensitivity/specificity (95.1%/89.9%) but a 17.7% finite-pool Tier A violation frequency at $n^+=200$, compared with 1.2% at seed 42. The audit reuses the same finite image pool and does not verify patient independence, so the difference motivates caution and does not validate or disprove the mathematical population guarantee. Two seeds are not enough for a multi-seed robustness claim.

Across seeds 42 and 43, source-only LoRA reached B0 sensitivity/specificity of 95.7%/88.8% and 95.1%/89.9%, respectively. At $n^+=200$, the image-pool Tier A violation frequencies were 1.2% and 17.7% over 1,000 repetitions. Two seeds are not enough to establish robustness, and these finite-pool results neither validate nor disprove population PAC coverage; patient linkage is unverified. The source split manifests, adapter weights, classifier heads, and complete run JSON files are linked below for reproduction.

### 3.7 Multi-seed and nested target-pool audit

Across 10 source-head seeds, RAD-DINO TBX11K sensitivity averaged 92.6% (SD 2.1%) for logistic regression and 94.6% (SD 1.8%) for the MLP; corresponding Shenzhen means were 82.3% (SD 2.6%) and 85.2% (SD 2.2%). ResNet50-A1 TBX11K means were 87.6% (SD 3.6%) and 87.1% (SD 3.8%), and Shenzhen means were 77.4% (SD 1.9%) and 71.8% (SD 3.8%). These heads used 639 source-training images for TBX11K and 6,505 for Shenzhen.

Across both encoders, targets, and heads, the 1,000-repetition nested image-pool audit produced Tier A violation frequencies of 4.7–6.7% at $n^+=29$ (the binomial floor benchmark is 4.71%) and 2.9–6.9% at $n^+=100$ (benchmark 2.37%). CRC per-repetition holdout violation frequencies ranged from 21.8% to 48.6% over the audited grid; these are not CRC type-I error rates because CRC controls marginal expected risk. The empirical target pools were finite, calibration/evaluation halves reused the same observed positives across repetitions, and patient linkage is unverified; therefore none of these frequencies validates patient-i.i.d. PAC coverage. Full results and intervals are in [`ResNet50-A1`](../experiments/runs/robustness_audits/general-767268603ca0/results.json) and [`RAD-DINO`](../experiments/runs/robustness_audits/chest_xray-110cbc18d513/results.json).

The WSR null simulation rejected 50 of 2,000 null hypotheses (2.5%; 95% CP interval 1.86–3.28%) under iid Bernoulli losses. Cross-fitted density-ratio diagnostics had effective sample sizes of 415/660 (ResNet50-A1) and 447/660 (RAD-DINO) for TBX11K, and 43/336 and 69/336 for Shenzhen. These odds are diagnostics, not verified shift weights or known upper bounds.

### 3.8 Obsolete/unresolved-label sensitivity

The extreme remappings changed RAD-DINO's logistic Tier A violation rate at $n^+=200$ from 0.9% (all ambiguous records non-active) to 8.5% (all active); the MLP rates were 0.3% and 0.6%. For ResNet50-A1, the corresponding ranges were 0.3–0.9% for logistic regression and 0.0–0.1% for the MLP. These are sensitivity analyses of two deliberately extreme label policies, not evidence that ambiguous findings are biologically positive or negative; see the [`ResNet50-A1`](../experiments/runs/label_sensitivity/general-767268603ca0/results.json) and [`RAD-DINO`](../experiments/runs/label_sensitivity/chest_xray-110cbc18d513/results.json) result files.

### 3.9 Selective-verification budget sweep

The 64-cell bootstrap simulation used $N=1{,}000$–6,496, prevalence 0.05–0.40, and $q=0.10$–1.00 with 100 repetitions per cell. Naive Tier A issued in every cell and its observed violation frequency ranged from 2% to 100%. Four-bound VB issued in 44/64 cells, with 0–2% violations among issued certificates; linear IPW-Hoeffding issued in 9/64 cells with no observed violations. These conditional finite-simulation frequencies do not establish future-population coverage; the full grid and Clopper–Pearson intervals are in the [`budget-sweep results`](../experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_8_budget_sweep/results.json).

## 4. Discussion

The exact order-statistic rule has a transparent refusal floor: with fewer
than 29 independent representative target-positive calibration units, the
specified $(\alpha,\delta)=(0.10,0.05)$ certificate is infeasible. Above
that floor, larger calibration samples lower mean sensitivity toward the
target and raise specificity; at the smallest feasible $n^+$, higher
sensitivity comes at the cost of lower specificity. This trend is descriptive
of the finite TBX11K holdout; repeated subsamples reuse a finite image pool
and do not establish PAC validity for patient populations.

On Shenzhen, the source threshold missed the 90% sensitivity target for both
heads (79.2% logistic and 77.4% MLP), despite specificities of 84.0% and
89.6%. This target-site result illustrates why a source operating point may
not transport. Tier A at $n^+=100$ restored mean MLP sensitivity to 92.1%, with
specificity 62.9% and a 4.9% empirical violation rate; logistic violation
was 28.8%. At this calibration size, specificities around 40--63% imply that
a substantial share of non-TB images in the holdout would be flagged for
follow-up: about 37--60 per 100 across the reported heads. This could increase
confirmatory-testing workload; the image-level holdout does not estimate
referral burden or predictive value in a deployment population. The 4.9%
violation rate and 4.9% frequency of specificity at least 70% are distinct
resampling measures that happen to be numerically equal.

On the primary TBX11K B0 comparison, both RAD-DINO heads exceeded the
corresponding ResNet50-A1 source-threshold sensitivity and specificity
(91.5%/64.3% and 95.7%/57.5% versus 84.1%/33.3% and 88.4%/31.0%). This is a
descriptive encoder comparison, not evidence of a statistically tested
advantage. In the N1-versus-N2 negative-spectrum control, MLP specificity rose
from 57.5% to 76.8% with healthy-only negatives, while logistic specificity
rose from 64.3% to 85.6%; the same positive holdout was used for each head.
These results reinforce that specificity depends on the negative mix.

The random-split control's source-threshold specificity (about 99%) was much
higher than the leave-site-out TBX11K values (about 58–64%). This is
consistent with site-mixed evaluation being easier, but split composition
differs and identity linkage is unknown, so the contrast is not a causal
estimate of site shift. The origin probe and negative-spectrum control
support caution when interpreting a single aggregate metric.

The nested audits add seed variability and target-pool resampling, but they do not repair the identity limitation: the 660/336 image-positive pools cannot be treated as verified patient-i.i.d. samples. The observed Tier A frequencies therefore show finite-pool behavior and motivate careful interpretation of the mathematical certificate's assumptions; they are not a re-estimate of population PAC coverage. The extreme-label audit also shows that decisions can depend on how TBX11K's 140 obsolete/unresolved cases are handled, particularly for RAD-DINO logistic regression.

Naive use of selectively verified positives performed poorly in the
simulation. The four-bound certificate traded utility for conservatism and
failed the pre-specified specificity-gap gate; the Hoeffding alternative
refused throughout. These findings support explicit refusal and motivate
better-designed verification, not an assertion that the evaluated VB bound
is practically adequate. The simulation uses fully observed labels to create
the selection process and cannot substitute for a prospective verification
study.

The ROI analysis is exploratory: box area is only a radiographic size proxy,
the annotations cover selected positive images only, and stratum counts are
22–23. No severity claim is justified. The LoRA arm remains an ablation; it
must not be used to make model-selection claims from target-validation
outcomes or to assert clinical advantage.

### Limitations and open research checkpoints

1. Patient linkage is unverified for all three datasets, so the image-level splits do not establish independent patient sampling.
2. The image-identity audit found 126 exact decoded-pixel duplicate groups within TBX11K, all in the non-TB class; 38 groups span its official train/validation lists. No exact cross-site duplicates were found. The perceptual-hash screen found 18 cross-site candidates, which require manual review and are not confirmed duplicates. Embedding-neighbor candidates are reported in the audit JSON.
3. Exact duplicates make the original site-mixed random split unsuitable as an unqualified control; the revised run excludes exact duplicate aliases before splitting.
4. The label reference standard and cohort spectrum are imperfect proxies for
   clinical confirmation and deployment populations.
5. Selective verification is simulated, not prospectively observed; the VB
   and linear arms are image-level audits.
6. The project Shenzhen ROI CSV contains 250 boxes across 68 images. The public archive has two reader-specific annotation collections, each covering 336 TB-positive images, but documentation does not explain the derivation of the project CSV from them.
7. Age, sex, and TB-history strata are absent from the manifest, so Mondrian
   stratification could not be evaluated.
8. Ethics approval or waiver, dataset terms, and data-use permissions have
   not been documented as complete; resolve them before submission. Authority
   to redistribute derived artifacts also remains outstanding, and nothing
   is released by these commands.

### Conclusion

This work implements an exact target-positive certify-or-refuse threshold,
audits its image-level trade-off, and compares it with selective-verification
procedures. The 29-positive floor is exact for the selected error and
confidence levels, but the empirical studies do not establish all distributional and labeling
assumptions. Future claims require prospective cohorts,
pre-specified verification, ethics approval or waiver, completed literature
screening, and data-use approval.

## Reproducibility and result evidence

Run these commands from the project root after installing `uv` and placing
the public source data in `data/raw/`. Source data are not included.

```powershell
uv sync --extra vision
uv run tb-certify inspect-data --project-root .
uv run tb-certify preprocess-data --project-root . --image-size 224
uv run tb-certify preprocess-data --project-root . --output-dir data/processed/rad_dino_518 --image-size 518
uv run --extra vision tb-certify extract-embeddings --project-root . --encoder general --output-dir artifacts/embeddings/general-767268603ca0
uv run --extra vision tb-certify extract-embeddings --project-root . --encoder chest_xray --output-dir artifacts/embeddings/chest_xray-110cbc18d513
uv run python scripts/probe_dataset_origin.py --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --identity-audit artifacts/audits/image_identity.json --output experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_4_origin_probe_deduped.json --seed 42 --folds 5
uv run python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/general-767268603ca0 --output-dir experiments/runs/primary_tbx11k_loso/general-767268603ca0 --target-site TBX11K --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513 --target-site TBX11K --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_8_selective --target-site TBX11K --repetitions 1000 --selective-repetitions 100 --seed 42
uv run --extra vision python scripts/run_primary_real_study.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/secondary_shenzhen_loso/chest_xray-110cbc18d513 --target-site Shenzhen --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_random_split_control.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/random_image_control/chest_xray-110cbc18d513 --repetitions 1000 --seed 42
uv run --extra vision python scripts/analyze_lesion_size.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output experiments/runs/secondary_shenzhen_loso/chest_xray-110cbc18d513/p0_9_lesion_size/results.json
uv run --extra vision python scripts/run_lora_ablation.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p1_2_lora_final --epochs 3 --batch-size 1 --gradient-accumulation 4 --repetitions 1000 --seed 42
uv run --extra vision python scripts/run_lora_ablation.py --project-root . --embedding-dir artifacts/embeddings/chest_xray-110cbc18d513 --output-dir experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p1_2_lora_seed43 --epochs 3 --batch-size 1 --gradient-accumulation 4 --repetitions 1000 --seed 43
uv run --extra vision python scripts/render_result_thumbnails.py --output-dir paper/figures
```

### Linked last-run evidence

- Primary B0/Tier A: [`results.json`](../experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/results.json)
- ResNet50-A1 baseline: [`results.json`](../experiments/runs/primary_tbx11k_loso/general-767268603ca0/results.json)
- Site-origin probe: [`p0_4_origin_probe_deduped.json`](../experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_4_origin_probe_deduped.json)
- Selective verification: [`results.json`](../experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_8_selective/results.json)
- Linear IPW--Hoeffding gate: [`results.json`](../experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p0_8_p1_linear_selective/results.json)
- Shenzhen target: [`results.json`](../experiments/runs/secondary_shenzhen_loso/chest_xray-110cbc18d513/results.json)
- Random split: [`results.json`](../experiments/runs/random_image_control/chest_xray-110cbc18d513/results.json)
- ROI tertiles: [`results.json`](../experiments/runs/secondary_shenzhen_loso/chest_xray-110cbc18d513/p0_9_lesion_size/results.json)
- LoRA seed 42: [`results.json`](../experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p1_2_lora_final/results.json), [`adapter`](../experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p1_2_lora_final/adapter/), [`classifier head`](../experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p1_2_lora_final/classifier_head.pt), [`source split`](../data/splits/lora_source_seed42.csv)
- LoRA seed 43: [`results.json`](../experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p1_2_lora_seed43/results.json), [`adapter`](../experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p1_2_lora_seed43/adapter/), [`classifier head`](../experiments/runs/primary_tbx11k_loso/chest_xray-110cbc18d513/p1_2_lora_seed43/classifier_head.pt), [`source split`](../data/splits/lora_source_seed43.csv)
- Encoder overlap: [`audit JSON`](../artifacts/audits/rad_dino_training_overlap_110cbc18.json)
- Split manifests: [`TBX11K primary`](../data/splits/), [`random control`](../data/splits/random_image_control_seed42.csv)

Last preview regeneration uses the aggregate JSON artifacts above. It reports
rates and confidence intervals only, not individual records. The SVG files
are views of those JSONs and are not independent evidence.

## Selected references

This list supports claims in the working draft. Literature searching was
exploratory in PubMed and IEEE Xplore;

1. Qin, Z. Z. et al. Using artificial intelligence to read chest radiographs
   for tuberculosis detection: a multi-site evaluation of the diagnostic
   accuracy of three deep learning systems. *Scientific Reports* 9, 15000
   (2019). [doi:10.1038/s41598-019-51503-3](https://doi.org/10.1038/s41598-019-51503-3).
2. Tavaziva, G. et al. Chest X-ray analysis with deep learning-based software
   as a triage test for pulmonary tuberculosis: an individual patient data
   meta-analysis of diagnostic accuracy. *Clinical Infectious Diseases* 74,
   1390–1400 (2022). [doi:10.1093/cid/ciab639](https://doi.org/10.1093/cid/ciab639).
3. Bates, S. et al. Distribution-free, risk-controlling prediction sets.
   *Journal of the ACM* 68(6), 43:1–43:34 (2021).
   [doi:10.1145/3478535](https://doi.org/10.1145/3478535).
4. Zecchin, M. et al. Generalization and informativeness of weighted
   conformal risk control under covariate shift. arXiv:2501.11413 (2025).
   [arXiv:2501.11413](https://arxiv.org/abs/2501.11413).
5. Almeida, D. C. et al. High probability risk control under covariate shift.
   *Proceedings of Machine Learning Research* 266, 133–152 (2025).
   [PMLR 266](https://proceedings.mlr.press/v266/almeida25a.html).
6. Pérez-García, F. et al. Exploring scalable medical image encoders beyond
   text supervision. *Nature Machine Intelligence* 7, 119–130 (2025).
   [doi:10.1038/s42256-024-00965-w](https://doi.org/10.1038/s42256-024-00965-w).

# License and Citation Dataset

## Montgomery and Shenzhen dataset citations

The [Montgomery County CXR](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/NLM-MontgomeryCXRSet-ReadMe.pdf)
and [Shenzhen Hospital CXR](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/NLM-ChinaCXRSet-ReadMe.docx)
readmes cite the following papers:

1. Candemir S, Jaeger S, Musco J, Xue Z, Karargyris A, Antani SK, Thoma GR,
   Palaniappan K. Lung segmentation in chest radiographs using anatomical
   atlases with nonrigid registration. *IEEE Transactions on Medical Imaging*
   33(2), 577–590 (2014). [doi:10.1109/TMI.2013.2290491](https://doi.org/10.1109/TMI.2013.2290491). PMID: 24239990.
2. Jaeger S, Karargyris A, Candemir S, Folio L, Siegelman J, Callaghan F, Xue
   Z, Palaniappan K, Singh RK, Antani S, Thoma G, Wang YX, Lu PX, McDonald CJ.
   Automatic tuberculosis screening using chest radiographs. *IEEE
   Transactions on Medical Imaging* 33(2), 233–245 (2014).
   [doi:10.1109/TMI.2013.2284099](https://doi.org/10.1109/TMI.2013.2284099). PMID: 24108713.

## [TBX11K](https://github.com/yun-liu/Tuberculosis#citation) 

```bib
@article{liu2023revisiting,
  title={Revisiting Computer-Aided Tuberculosis Diagnosis},
  author={Liu, Yun and Wu, Yu-Huan and Zhang, Shi-Chen and Liu, Li and Wu, Min and Cheng, Ming-Ming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={46},
  number={4},
  pages={2316--2332},
  year={2024},
  publisher={IEEE},
  doi={10.1109/TPAMI.2023.3330825}
}

@inproceedings{liu2020rethinking,
  title={Rethinking Computer-Aided Tuberculosis Diagnosis},
  author={Liu, Yun and Wu, Yu-Huan and Ban, Yunfeng and Wang, Huifang and Cheng, Ming-Ming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={2646--2655},
  year={2020},
  doi={10.1109/CVPR42600.2020.00272}
}
```

_Updated 23 October 2026._
