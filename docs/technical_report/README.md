# Technical Report Artifacts

This directory contains the arXiv-oriented technical report for the
Qwen3-1.7B expressivity-engineering study.

- `expressivity_engineering_qwen3_1p7b.tex`: self-contained LaTeX source.
- `references.bib`: bibliography used by the report.
- `expressivity_engineering_qwen3_1p7b.pdf`: rendered report.
- `generated/corrected_master_table.tex`: native LaTeX full-results table.
- `../../scripts/render_qwen3_1p7b_math_ood_report_table_tex.py`: report-table
  renderer; it reads the authoritative consolidated JSON and writes the LaTeX
  fragment plus a provenance manifest.

The consolidated numeric source of truth remains
`docs/experiment_records/2026-07-18_qwen3-1p7b-math-ood-corrected-consolidated-master-table.md`.
The report summarizes those records; it does not replace them.

Appendix A condenses the post-study research agenda from
`../followups/2026-07-31_expressivity-engineering-follow-up.md`. The standalone
note remains the complete provenance record; the appendix is a shorter formal
summary and explicitly labels every proposal as retrospective and unvalidated.
Its coverage includes the TriGLU/LatentMoE bottleneck connection, structured
SHS HyperNetworks, additive and multiplicative expert composition, corrected
dynamic-SwiGLU complexity, attention alternatives, frequency-shaped EE,
recurrence/unrolling, phase-locked attention, and the required falsification
matrix.

Regenerate the full-results table from the authoritative JSON, then build with
Tectonic:

```bash
python scripts/render_qwen3_1p7b_math_ood_report_table_tex.py
cd docs/technical_report
tectonic -X compile expressivity_engineering_qwen3_1p7b.tex
```

After rebuilding, render every PDF page to PNG and inspect the appendix
transition, equations, table, bibliography, page numbering, and references.
