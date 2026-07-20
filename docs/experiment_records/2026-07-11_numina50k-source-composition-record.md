# 2026-07-11 NuminaMath 50K Source Composition Record

Date: 2026-07-11

## Scope

This record summarizes the exact source-category composition of the committed
50,000-row training ledger and 100-row validation ledger for
`numina_math_cot_50k_decontam_v3`. Counts come from `selected_rows.tsv`; no
training or evaluation artifact was modified.

## Training Rows

| Source category | Rows | Share |
|---|---:|---:|
| `cn_k12` | 16,691 | 33.382% |
| `orca_math` | 8,918 | 17.836% |
| `olympiads` | 8,859 | 17.718% |
| `synthetic_math` | 8,787 | 17.574% |
| `synthetic_amc` | 3,967 | 7.934% |
| `aops_forum` | 1,882 | 3.764% |
| `math` | 454 | 0.908% |
| `amc_aime` | 227 | 0.454% |
| `gsm8k` | 215 | 0.430% |
| Total | 50,000 | 100.000% |

Useful source-level groupings are:

- broad K-12 proxy (`cn_k12`): 33.382%;
- explicitly GSM8K: 0.430%;
- mixed general/synthetic (`orca_math`, `synthetic_math`, `math`): 36.318%;
- explicitly competition-heavy (`olympiads`, `synthetic_amc`, `aops_forum`,
  `amc_aime`): 29.870%.

These are provenance groupings, not audited difficulty labels. `cn_k12` spans a
wide range rather than meaning elementary-only, while the mixed categories can
contain both basic and difficult problems. Therefore the ledger establishes
that elementary-style material is present, but it cannot identify its exact
share without a row-level difficulty audit.

## Validation Rows

| Source category | Rows | Share |
|---|---:|---:|
| `cn_k12` | 36 | 36% |
| `synthetic_math` | 21 | 21% |
| `orca_math` | 19 | 19% |
| `olympiads` | 13 | 13% |
| `synthetic_amc` | 3 | 3% |
| `aops_forum` | 3 | 3% |
| `gsm8k` | 3 | 3% |
| `amc_aime` | 1 | 1% |
| `math` | 1 | 1% |

The 100-row validation set is directionally similar but too small for precise
source-level estimates.

## Loss-Weighting Caveat

Row share is not necessarily the effective SFT-loss share. The trainer computes
loss over assistant tokens, so categories with longer reasoning traces can
contribute more tokens and gradient mass per row. Competition and olympiad
solutions are often longer than elementary arithmetic solutions. A tokenized
source audit is required before claiming that one difficulty regime dominates
the optimization objective.

This composition helps explain why lower held-out SFT loss need not imply a
uniform gain on GSM8K, MATH-500, and OlympiadBench: the validation objective is a
mixture of source and length regimes, while each benchmark measures a narrower
external distribution with exact-answer scoring.
