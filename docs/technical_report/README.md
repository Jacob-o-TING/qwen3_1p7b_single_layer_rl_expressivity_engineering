# Technical Report Artifacts

This directory contains the arXiv-oriented technical report for the
Qwen3-1.7B expressivity-engineering study.

- `expressivity_engineering_qwen3_1p7b.tex`: self-contained LaTeX source.
- `references.bib`: bibliography used by the report.
- `expressivity_engineering_qwen3_1p7b.pdf`: rendered report.
- `figures/`: report-local copies of publication figures.

The consolidated numeric source of truth remains
`docs/experiment_records/2026-07-18_qwen3-1p7b-math-ood-corrected-consolidated-master-table.md`.
The report summarizes those records; it does not replace them.

To build with Tectonic:

```bash
tectonic -X compile expressivity_engineering_qwen3_1p7b.tex
```
