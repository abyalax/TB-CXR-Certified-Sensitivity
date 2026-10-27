#import "@preview/juti:0.1.1"
#import "setup.typ": setup

#let authors = juti.init-authors((
  (
    name: "Abya",
    institution-ref: 0,
    contribution-refs: (0, 1, 2, 3, 4, 5, 7, 8, 10),
  ),
))

#let institutions = (
  (
    name: "Independent Researcher",
    address: "Indonesia",
  ),
)

#show: juti.template.with(
  title: "Certify-or-refuse sensitivity calibration for TB chest-radiograph triage at new sites",
  authors: authors,
  corresponding-ref: 0,
  corresponding-email: "abyalaxx@gmail.com",
  institutions: institutions,
  abstract: [
    Chest-radiograph artificial intelligence for tuberculosis (TB) triage needs
    a decision threshold that may not transfer across sites. We implement a
    certify-or-refuse procedure that calibrates a frozen score on target-site
    TB-positive records and issues a threshold only when an exact,
    finite-sample class-conditional sensitivity criterion is feasible. The
    target false-negative rate is 0.10 and the failure probability is 0.05,
    yielding a 29-positive certification floor. Frozen ResNet50 and RAD-DINO
    embeddings were paired with source-only logistic and small multilayer
    perceptron heads on Montgomery, Shenzhen, and TBX11K public chest X-ray
    data. Each image represents one patient in the three datasets, so the results
    are exploratory patient-level analyses. The PAC assumptions and clinical
    validity are not established by this one-to-one mapping. On the TBX11K holdout, the RAD-DINO source threshold achieved
    91.5% sensitivity and 64.3% specificity for logistic regression, and
    95.7% and 57.5% for the multilayer perceptron. A site-origin probe achieved
    98.0% balanced accuracy. Simulated selective verification showed that
    naive calibration on selectively confirmed positives can violate the
    sensitivity target, whereas a conservative verification-bias certificate
    refused or reduced specificity. The study is an implementation and audit,
    not a validation of the PAC assumptions or a clinical-deployment claim.
  ],
  keywords: (
    "tuberculosis",
    "chest radiograph",
    "threshold calibration",
    "conformal risk control",
    "selective verification",
    "domain shift",
  ),
  meta: (
    received: "Not submitted",
    revised: "-",
    accepted: "-",
    online: "-",
    doi: none,
  ),
  bib: bibliography("references.bib"),
  ..setup,
)

= Introduction

Artificial-intelligence-assisted chest radiography can reduce the workload of
confirmatory TB testing, but one score threshold is not automatically portable
between screening populations. Multi-site evaluation and an individual-patient
meta-analysis both support local evaluation of operating thresholds
@melendez2020 @tavaziva2022. In field workflows, microbiological confirmation
may itself be selective. Low-score participants can then be underrepresented
among observed positives, so those positives are not a random target
calibration sample.

This study asks whether a target site can issue a finite-sample,
class-conditional sensitivity threshold, or explicitly refuse when the
available evidence is inadequate. Our contribution is an implementation and
empirical audit of exact target-only calibration, risk-control comparators, and
selective-verification certificates. It neither proposes a new theorem nor
establishes clinical deployment validity.

= Methods

== Data and protocol

Montgomery (138 images) and Shenzhen (662 images) use the
National Library of Medicine documented filename-label convention. TBX11K uses
only its official train and validation lists; an active-TB positive requires an
`ActiveTuberculosis` lesion annotation. Obsolete-only and unresolved findings
are excluded. The processed cohort contains 9,060 images: 138 Montgomery, 662
Shenzhen, and 8,260 TBX11K. Explicit patient identifiers may be blank, but dataset documentation and
project confirmation establish one image per patient; image-based splits are
therefore patient-level.

We froze an ImageNet ResNet50-A1 encoder and the RAD-DINO chest-radiograph
encoder, each pinned to an audited revision. Logistic-regression and small MLP
heads trained only on source sites. TBX11K is the primary target: official
training records provide target calibration and official validation records are
the patient-level evaluation holdout (one image per patient).

== Exact target-only certificate

Let higher scores indicate active TB. For target false-negative rate
$alpha = 0.10$, failure probability $delta = 0.05$, and $n^+$ target-positive
calibration records, define

#align(center)[
  $ k^* = max { k: P("Bin"(n^+, alpha) <= k) <= delta }. $
]

The threshold is the $(k^* + 1)$-st smallest positive score. If no such $k^*$
exists, the method returns *REFUSE*. The first feasible sample size is 29
positives. Validity requires representative independent target positives, a
fixed score rule, accurate reference labels, and stationarity. The one-image-per-patient sampling unit is confirmed. Certificate outputs
below remain empirical audit estimates and do not establish PAC assumptions.

== Comparators and selective verification

The software implements marginal conformal risk control @bates2021,
test-point-weighted CRC @zecchin2025, and importance-weighted learn-then-test
using a WSR betting p-value @almeida2025. Weighted methods require a valid,
pre-specified positive-conditional density ratio and an upper bound; neither is
asserted for the real-data study.

For the verification simulation, every record above a fixed reference score is
verified, while low-score records are independently verified with probability
$q$. Naive Tier A is compared with a four-bound Clopper--Pearson
verification-bias (VB) certificate on a source-defined fixed threshold grid. A
linear IPW--Hoeffding variant is evaluated as a tightening gate.

= Results and discussion

== Source shift

On the TBX11K holdout, the RAD-DINO logistic source threshold reached 91.5%
sensitivity and 64.3% specificity; the MLP reached 95.7% sensitivity and 57.5%
specificity. A five-fold site-origin probe over all RAD-DINO embeddings reached
99.3% accuracy, 98.0% balanced accuracy, and 99.9% macro one-vs-rest AUC. The
probe shows that embeddings retain site information; it does not identify its
cause or prove causal shortcutting.

#figure(
  table(
    columns: 3,
    align: center,
    table.header([*Head*], [*Sensitivity*], [*Specificity*]),
    [Logistic], [91.5%], [64.3%],
    [MLP], [95.7%], [57.5%],
  ),
  caption: [Source-threshold performance on the official TBX11K validation split using RAD-DINO. Results are exploratory patient-level estimates (one image per patient).],
) <tab:b0>

== Tier A and negative spectrum

The TBX11K target split contains 496 calibration positives; the holdout has
164 positives and 1,600 negatives. At $n^+ = 29$, RAD-DINO logistic Tier A had
mean sensitivity 96.0% and specificity 41.8%; at $n^+ = 200$, it had 93.2% and
58.8%. The MLP had 97.0%/39.3% at 29 and 94.5%/65.7% at 200. These finite
holdout, patient-level estimates illustrate the sensitivity--specificity cost of
the certificate; they do not validate all PAC assumptions.

At the logistic source threshold, specificity was 85.6% against healthy-only
TBX11K negatives but 64.3% against healthy plus non-TB disease. Any
specificity claim must state its negative spectrum.

== Verification-bias simulation

Fully labelled TBX11K training records were used only to simulate selective
verification, with 100 repetitions per policy. Naive Tier A issued a threshold
in every audit. At $q = 0.10$, its observed violation rate was 100% for both
heads; at $q = 0.25$, it was 100% for logistic regression and 77% for the MLP.
The four-bound VB certificate was conservative: its issue rate at $q = 0.10$
was 4% for logistic regression and 7% for the MLP, with no observed violations
among issued runs. It issued in every run at $q >= 0.25$, again with no observed
violations. The linear IPW--Hoeffding arm issued no threshold in these audits,
so it failed the pre-specified utility gate.

#figure(
  table(
    columns: 4,
    align: center,
    table.header([*Low-score verification $q$*], [*Method*], [*Issue rate*], [*Observed violation rate*]),
    [0.10], [Naive Tier A (logistic / MLP)], [100% / 100%], [100% / 100%],
    [0.10], [VB certificate (logistic / MLP)], [4% / 7%], [0% / 0%],
    [0.25], [Naive Tier A (logistic / MLP)], [100% / 100%], [100% / 77%],
    [0.25], [VB certificate (logistic / MLP)], [100% / 100%], [0% / 0%],
  ),
  caption: [Selective-verification simulation on fully labelled TBX11K training records. The labels were used only to simulate the verification policy.],
) <tab:selective>

The experiment demonstrates why selectively confirmed positives cannot be
naively treated as a random calibration sample. The exact target-only rule
makes the 29-positive evidence floor explicit and refuses below it. Clinical
use is not established: one image per patient is confirmed, but selective verification was simulated rather
than prospectively deployed. A prospective study should lock the model,
collect patient-level records, and follow an auditable confirmation policy.

= Conclusion

This repository provides a reproducible patient-level audit (one image per patient) of a
certify-or-refuse sensitivity-calibration workflow for TB chest-radiograph
triage. The frozen RAD-DINO analyses show material site information and a
negative-spectrum dependence of specificity. Under simulated selective
verification, naive calibration can fail the 90% sensitivity objective whereas
the conservative VB procedure protects the observed audit at a substantial
cost in refusals or specificity. Prospective patient-level validation remains
the required next step.

#set heading(numbering: none)

= CRediT authorship contribution statement

#juti.credits(authors)

= Declaration of competing interest

The author declares no known competing financial interests or personal
relationships that could have appeared to influence the work reported here.

= Acknowledgement

No external funding was received for this work.

= Data availability

The analysis uses public datasets subject to their respective terms. Raw
images, derived embeddings, score vectors, and patient-identifying material
are not redistributed with this repository. Split manifests, configuration,
and code are retained locally for reproducibility.

= Declaration of generative AI and AI-assisted technologies in the writing process

Generative AI was used to improve drafting clarity and to assist conversion of
the manuscript to Typst. The author reviewed and remains responsible for the
final manuscript.
