# JUTI manuscript

This is the active manuscript project for *Certify-or-Refuse Sensitivity
Calibration for TB Chest-Radiograph Triage at New Sites*. It uses the official
Juti Typst template, version `@preview/juti:0.1.1`.

## Files

- `main.typ` is the submission source.
- `setup.typ` contains issue and page metadata that should be finalized by the
  journal editor at acceptance.
- `references.bib` contains the bibliography cited by `main.typ`.
- `manuscript.md` is the pre-conversion Markdown draft, retained only for
  traceability.
- `literature_search.md` records the non-systematic search log.

## Build

From the repository root:

```powershell
.\paper\build.ps1
```

The script first uses `typst` on `PATH`; otherwise it discovers a portable
`typst.exe` below `paper/.tools/typst/`. It writes the ignored output
`paper/main.pdf`. During editing, run:

```powershell
.\paper\build.ps1 -Watch
```

The local setup uses Typst 0.15.1. The first build downloads the pinned Juti
package and its `numbly` dependency into Typst's normal package cache. Juti
prefers Liberation Serif; the current Windows environment does not provide that
font, so Typst emits a non-fatal font warning and renders with its available
fallback. Install Liberation Serif system-wide if an exact template font match
is required for final submission.

## Submission checks

Before submission, replace the provisional affiliation and corresponding email
if needed, confirm every BibTeX entry against its primary source, set official
issue, page, DOI, and article-history metadata, and review the data-availability
and generative-AI declarations. The paper intentionally describes all current
numeric results as exploratory patient-level analyses (one image per patient);
the PAC assumptions and clinical validity are not thereby established.
