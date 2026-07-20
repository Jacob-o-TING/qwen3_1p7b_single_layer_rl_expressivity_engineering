# Step-294 OOD Subject And Code Failure Audit / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-17_step294-ood-subject-and-code-failure-audit-record.md](../2026-07-17_step294-ood-subject-and-code-failure-audit-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **Step-294 OOD Subject And Code Failure Audit**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


Date: 2026-07-17

Status: **COMPLETE for the available TriGLU step-294 predictions.** This was a
read-only audit plus cached-prediction code review. No checkpoint was changed,
and no model generation was repeated.

## Score Corrections And Monitor Contract

The historical HumanEval+ `0/164` and LiveCodeBench `0/1055` values were
evaluator-confounded rather than valid model scores:

- HumanEval+ after the conservative parser and fixed sandbox: `36/164 =
  21.951%`;
- LiveCodeBench after repairing the sandbox `output` contract and reviewing the
  exact frozen six-shard predictions: `107/1055 = 10.142%`;
- MBPP historical report, held fixed for parser sensitivity: `146/500 =
  29.200%`.

The human-readable monitor now reports HumanEval+ and CodeAvg before and after
the HumanEval+ parser/sandbox recovery. To isolate that recovery, corrected
LiveCodeBench and historical MBPP are held fixed in both cells:

```text
CodeAvg_pre  = (0/164 + 146/500 + 107/1055) / 3 = 13.114%
CodeAvg_post = (36/164 + 146/500 + 107/1055) / 3 = 20.431%
```

This is an equal-benchmark average, not a sample-count-weighted accuracy. The
raw historical CodeAvg remains invalid because it includes evaluator failures.
Accordingly, the effective `code_mean`, eight-benchmark OOD mean, and
three-category OOD mean use the corrected HumanEval+ and LiveCodeBench scores;
the invalid zeroes remain visible only as provenance and in the explicitly
labeled pre-recovery comparison.

## GPQA-Diamond Correct-Answer Audit

All `198/198` predictions mapped uniquely to raw dataset metadata. GPQA-Diamond
is a four-choice expert natural-science benchmark requiring a final A-D letter,
not a pure-math free-response benchmark. TriGLU answered `59/198 = 29.798%`:

| Domain | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Physics | 29 | 86 | 33.72% |
| Chemistry | 26 | 93 | 27.96% |
| Biology | 4 | 19 | 21.05% |

The correct rows span organic chemistry (20), quantum mechanics (12), general
chemistry (6), general physics (6), molecular biology (4), electromagnetism and
photonics (3), relativistic mechanics (3), astrophysics (2), high-energy
particle physics (2), and condensed-matter physics (1). The result therefore
reflects broad science knowledge and reasoning rather than a hidden math-only
subset.

## MMLU-Pro Correct-Answer Audit

MMLU-Pro uses ten-choice A-J questions with subject-specific few-shot examples
and a final-choice extraction contract. TriGLU answered `3943/12032 = 32.771%`.
Correct answers occur in every subject:

| Subject | Correct / Total | Accuracy |
|---|---:|---:|
| Biology | 371 / 717 | 51.74% |
| Business | 224 / 789 | 28.39% |
| Chemistry | 386 / 1132 | 34.10% |
| Computer science | 120 / 410 | 29.27% |
| Economics | 389 / 844 | 46.09% |
| Engineering | 216 / 969 | 22.29% |
| Health | 259 / 818 | 31.66% |
| History | 78 / 381 | 20.47% |
| Law | 197 / 1101 | 17.89% |
| Math | 627 / 1351 | 46.41% |
| Other | 255 / 924 | 27.60% |
| Philosophy | 122 / 499 | 24.45% |
| Physics | 320 / 1299 | 24.63% |
| Psychology | 379 / 798 | 47.49% |

The unexpectedly strong aggregate is consequently broad cross-domain
multiple-choice performance, not a score carried only by mathematics.

## HumanEval+ Failure Modes

The corrected cached review contains:

| Failure mode | Rows | Interpretation |
|---|---:|---|
| Passed | 36 | Functional solution passed expanded tests |
| No valid Python after conservative parsing | 42 | Generation/template collapse; no implementation to recover |
| Runtime `NameError` | 59 | Mostly stray Hebrew response labels; not Python grammar errors |
| Hidden-test `AssertionError` | 21 | Syntactically valid but semantically wrong implementation |
| Other runtime exceptions | 6 | ValueError 4, IndexError 1, AttributeError 1 |

Of the 59 `NameError` rows, 55 reference the Hebrew token `\u05ea\u05e9\u05d5\u05d1\u05ea`, two
reference an undefined `math`, one references `\u05ea\u05db\u05e0\u05d9\u05ea`, and one references
`umably`. None is a missing required entry point. HumanEval+'s dominant failure
is generation/template collapse followed by semantic logic errors, not ordinary
syntax mistakes.

## MBPP Failure Modes

The historical `146/500 = 29.2%` report decomposes primarily into wrong-task
selection and semantic errors, not grammar:

- 207 rows omit the requested function or extract an earlier demonstration
  function. One exact example requests `pos_count` but emits
  `similar_elements`, a few-shot repetition/task-selection failure rather than
  parser fabrication;
- 110 rows reach tests but fail assertions;
- 36 rows originally report runtime exceptions with the target present;
- only one row is a syntax/grammar failure.

Seventeen of the runtime rows were additionally confounded by the historical
OpenBLAS thread limit. Frozen-program replay under the repaired sandbox recovers
four passes and leaves four assertion failures plus nine genuine runtime errors.
This gives a targeted corrected lower bound of `150/500 = 30.0%`, with adjusted
taxonomy 150 pass, 207 missing/wrong implementation, 114 semantic assertion
failure, 28 runtime failure, and one syntax failure. It is not labeled a full
MBPP rescore because only the 17 known infrastructure rows were replayed.

## Scientific Boundary

The LiveCodeBench all-zero artifact and the MBPP wrong-task outputs are distinct
phenomena. LiveCodeBench had valid generated programs whose success text was
discarded by an evaluator field mismatch. MBPP genuinely contains many outputs
for the wrong function. Correcting evaluator contracts must not turn those
model failures into passes.

## 继承待办 / Pending Obligations Carried Forward
Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`.

- **PENDING-01 Eval Parity Matrix:** still pending. This audit repairs known
  evaluator defects but does not complete the full HF-versus-vLLM matrix.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deferred; no architecture dtype was
  changed.
- **PENDING-03 Registered SHS CausalLM Route:** deferred; this record concerns
  the existing TriGLU OOD checkpoint only.
