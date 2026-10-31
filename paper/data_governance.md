# Data Governance and Ethics Review

Status reviewed 31 October 2026 for this exploratory secondary analysis. This is a research-governance record, not legal advice, an institutional ethics determination, or permission to redistribute any source or derived data.

## Source data and licenses

- Montgomery and Shenzhen images/readings are distributed by the U.S. National Library of Medicine. The local NLM README files request that the data not be shared outside the research group or organization. NLM’s current dataset page describes these de-identified images and annotations as reviewed and exempted by the source institutions’ ethics boards and requests citation of the dataset and annotation papers ([NLM dataset page](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayDatasets.html)). That source-level statement does not determine whether this project requires a separate local ethics review, and it does not override the local README’s redistribution restriction.
- The TBX11K archive has no standalone license file in the downloaded ZIP. Project metadata records CC BY-NC-SA 2.0 for the annotation manifest; this record does not establish that every image, label, or downstream artifact has an identical license. Preserve the dataset and annotation citations and obtain rights-holder clarification before distributing derived artifacts.
- The encoder metadata records Apache-2.0 for `timm/resnet50.a1_in1k` and MIT for `microsoft/rad-dino`. Those model licenses do not grant rights to redistribute source images or project-derived outputs subject to dataset terms.

## Ethics determination

The analysis uses existing de-identified public datasets and does not contact participants. NLM’s source-level ethics statement is recorded above. No local institutional approval, exemption, or “not human subjects research” determination is documented in this repository. Obtain the appropriate determination from the responsible institution and report its exact status before submission; do not describe the project itself as approved or exempt based only on the source-dataset page.

## Derived outputs and release decision

This checkout stores embeddings, image-derived scores, model outputs, manifests, split assignments, and aggregate results. The applicable data terms do not clearly settle redistribution of these derived artifacts, especially for Montgomery/Shenzhen. Do not publish or transfer embeddings, per-image scores/predictions, split-level image records, or other image-linked outputs until each source’s terms and the rights holder’s position are documented. Aggregate statistics may also require review for attribution and contractual restrictions. Current repository artifacts remain local research materials; a public release is not authorized by this audit.

## Actions before submission or release

- Obtain and record the local institutional ethics determination.
- Confirm the scope of the TBX11K CC BY-NC-SA 2.0 notice for images, labels, annotations, and derived data with the dataset rights holder.
- Confirm NLM terms for derived embeddings, per-image scores, manifests, and aggregates; retain source attribution and citations.
- Review the pinned encoder licenses and their upstream model cards at the time of release.
- Publish only code and outputs for which reuse and redistribution permissions are documented; keep restricted source inputs out of the repository.
