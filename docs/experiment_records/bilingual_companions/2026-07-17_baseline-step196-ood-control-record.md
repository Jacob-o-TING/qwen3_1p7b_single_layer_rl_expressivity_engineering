# 2026-07-17 Baseline Step-196 OOD Control Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-17_baseline-step196-ood-control-record.md](../2026-07-17_baseline-step196-ood-control-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-17 Baseline Step-196 OOD Control Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


Date: 2026-07-17

Status: **COMPLETE.** All eight benchmark stages completed under the six-GPU
staged evaluator. The evaluator screen stopped and all six GPUs were idle at
the completion check. The directory-valued pre-baseline interposer remained in
place through the subsequent HumanEval+ prompt-protocol reruns and was released
only after those reruns also completed exactly.

## Fixed Identity

- Run ID: `qwen3_1p7b_ood_6x5090_baseline_step196_20260717_v1`.
- Model label: `baseline_step196`.
- Screen: `qwen_baseline_step196_ood6_20260717_v1`.
- Model source: durable `baseline_step_196` export.
- Protocol: vLLM greedy pass@1, seed `20260707`, response cap `3072`, six
  independent TP=1 replicas.
- Stage order: GPQA-Diamond, MMLU-Pro, HumanEval+, MBPP, C-Eval, IFEval, MGSM,
  then LiveCodeBench.

The existing GRPO pre-baseline barrier remains held until this control writes
its full OOD completion marker. No training update can race this evaluation.

## First Attempt And Repair

All six first-attempt GPQA workers completed, but their receipts merged to only
`188/198` unique identities. The merger correctly refused to publish a final
score. The failed stage is retained as
`reasoning_gpqa_staged6_failed_payloadhash_20260717_v1`.

The original ownership hash included the rank-local formatted sample payload.
Formatting after independent worker bootstrap can consume different RNG state,
so one canonical dataset index could compute different owners on different
ranks. This produced missing ownership even though every worker was healthy.

The repaired protocol assigns ownership only from the immutable coordinate
`(dataset, subset, canonical sample_index)`. Each receipt separately records a
`formatted_sample_sha256` integrity digest. The change preserves prompt-payload
evidence while making ownership independent of rank-local formatting RNG. It
does not change dataset exposure, seed, prompt, decoding, model weights, or
expected sample count.

Focused tests passed `4/4`. The recovered GPQA merge has exact `198/198`
coverage, zero duplicate identities, and final score `22.727%`.

## Final Results

| Category | Benchmark | Score | Coverage |
|---|---|---:|---:|
| Reasoning | GPQA-Diamond | 22.727% | 198/198 |
| Reasoning | MMLU-Pro | 34.533% | 12032/12032 |
| Code | HumanEval+ | 31.097% | 164/164 |
| Code | MBPP | 28.202% | 500/500 |
| Language | C-Eval | 45.692% | 1346/1346 |
| Language | IFEval | 27.046% | 477 reviewed rows from the 541-row dataset adapter |
| Language | MGSM | 59.017% | 2750/2750 |
| Code | LiveCodeBench | 9.764% | 1055/1055 |

The equal-weight category means are Code `23.021%`, Reasoning `28.630%`, and
Language `43.918%`. The unweighted eight-benchmark mean is `32.260%`; the
unweighted mean of the three category means is `31.856%`.

These are current all-model vLLM chat-protocol results. HumanEval+ paper parity
is not claimed. The untuned-base HumanEval+ collapse and the proposed prompt
matrix are documented separately in
`2026-07-17_humanevalplus-paper-anchor-protocol-mismatch-record.md`.

Compact metrics:
`compact_metrics/2026-07-17_baseline_step196_ood_control.json`.

The unexpectedly large MGSM separation from TriGLU step 294 was subsequently
audited row by row at the operational-feature level. The paired-language
counts, four observed failure modes, repetition statistics, and trace examples
are recorded in
`2026-07-17_triglu-mgsm-failure-mode-audit-record.md`.

## Barrier Closeout Amendment

After TriGLU OOD, untuned-base OOD, this baseline-step196 control, and the
full-164 corrected HumanEval+ cells for untuned base, TriGLU step-294, and
baseline step-196 all had durable completion markers, the empty directory-valued
`PRE_BASELINE_OOD_COMPLETE` interposer was atomically replaced by a regular
completion file. The existing GRPO controller released its `-f` wait without a
restart and resumed baseline from step 196 toward target 226 under
`resume_mode=auto`. No evaluator or training process contended for a GPU at the
handoff.

## Human-Readable Monitoring

The unified GRPO monitor is the recommended complete status command:

```bash
bash scripts/monitor_triglu_priority_to294_then_baseline196_20260715_v1.sh
```

It embeds the OOD-specific stage view and presents corrected/current evaluation
results before heritage chat-protocol results. The standalone
`monitor_qwen3_1p7b_ood_6x5090_step294_20260716_v1.sh` remains available for
focused OOD diagnostics. Both paths show each baseline-step196 stage, six-shard
progress, partial weighted score and sample count before merge, and final score
plus exact coverage after merge. Partial evidence is explicitly labeled not
final.

## 继承待办 / Pending Obligations Carried Forward
Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** deferred; this is one vLLM protocol cell.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deferred; neither custom path is
  changed by this baseline control.
- **PENDING-03 Registered SHS CausalLM Route:** deferred; SHS is out of scope.
