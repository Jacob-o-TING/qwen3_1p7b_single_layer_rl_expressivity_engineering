# 2026-07-11 Baseline AMC Partial Snapshot Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-11_baseline-amc-partial-snapshot-record.md](../2026-07-11_baseline-amc-partial-snapshot-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-11 Baseline AMC Partial Snapshot Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


## 状态 / Status
This is a provisional, completion-biased snapshot of an actively written
evaluation. It is not a final benchmark score and must not be entered into the
primary comparison table.

## Snapshot

During the whole-layer baseline AMC Average@32 evaluation, a bounded read-only
inspection observed:

```text
complete review rows: 876 / 1280
EvalScope-correct rows: 111
partial accuracy: 111 / 876 = 12.671233%
last source-file byte: LF (decimal 10)
```

The source was the active baseline review JSONL under:

```text
runs/sft_ordered_20260711_sft50k_v1/evaluations/
  layer10_whole_layer_baseline/amc_average_at_32/20260711_141405/
  reviews/qwen3-1p7b-single-layer-sft/paper_amc23_main.jsonl
```

The inspection did not lock, copy over, truncate, rename, or modify the source
file. The file ended in a complete newline at the observation point, and the
EvalScope writer continued independently.

## Provisional Interpretation

The 12.67% partial baseline accuracy is slightly below the completed SHS primary
AMC score of 13.44%. If the final baseline result remains in this neighborhood,
it would weaken the hypothesis that SHS alone causes the AMC collapse. It would
instead increase the relative plausibility of:

- a common SFT-recipe effect;
- a common long-form reasoning or overthinking effect;
- sensitivity to the locally selected temperature-1.0 AMC sampling protocol;
- a prompt, tokenizer, generation, or evaluator mismatch relative to the
  unpublished paper evaluation protocol;
- combinations of these mechanisms.

This observation does not yet establish any of those explanations. The active
prefix is not an unbiased sample of the 1,280 responses. Autoregressive
microbatch evaluation tends to complete shorter responses before longer ones,
and completion order may correlate with problem, length, formatting, cap hits,
and correctness. The remaining 404 responses can move the final score.

## Acceptance Gate

Only the completed 1,280-row EvalScope report, followed by receipt verification
and optional semantic audit, may replace this provisional observation. At that
point this record should remain as a history of the evolving hypothesis, not as
the cited final baseline result.

## Completion Milestone

The bounded 16:16 +08:00 monitor confirmed that baseline AMC generation had
finished and the same serial evaluator had advanced to MATH-500, which was at
approximately 129 of 500 responses. The ordered screen remained healthy with no
reported errors. This milestone supersedes the snapshot as a progress indicator,
but the final AMC score still requires reading and verifying the completed
report before it enters the comparison table.

## Final Result And Local Pull

The completed baseline AMC artifact was subsequently packaged remotely and
pulled to the local workspace. Remote and local archive SHA-256 matched:

```text
archive: baseline_amc_average_at_32_20260711_141405.tar.gz
archive SHA-256: 911c700aab6827955f817650568229576da9239ff551a6640e73f6133bd631eb
local root: audit_inputs/baseline_amc_average_at_32_20260711_141405/
```

The archive contains the evaluation manifest, task config, eval log, HTML and
JSON reports, and all 1,280 prediction and review rows. Integrity checks found
exactly 1,280 unique review indices. The report and an independent review-row
recomputation agree:

```text
EvalScope displayed score: 13.67%
raw correct rows: 175 / 1280
raw accuracy: 13.671875%
report SHA-256: c7bfdcd61a01545f9a9c8861f5344c1e340e9d342d0b57c8a9bddc057c3078a5
predictions SHA-256: 5b5506d043c544078c2cab000856e9d8152fea1a6b0da84aceee02f18c8d6a89
reviews SHA-256: 54885ca2b0cac2a1cafd6b4208cc88629e978dab541285b218bdfc22a5ef3ffe
```

For the completed SHS AMC evaluation, the raw score is 172 of 1,280, or
13.4375%. The whole-layer baseline therefore exceeds SHS by only three correct
sampled responses:

```text
absolute difference: +0.234375 percentage points
```

This final result strongly weakens an SHS-specific explanation for the sampled
AMC collapse. Both architectures received the same SFT data/order/seed and both
perform nearly identically under the local temperature-1.0 Average@32 protocol.
The result instead raises the relative plausibility of a common SFT-recipe,
long-form reasoning, sampling calibration, or evaluator/protocol effect.

It does not distinguish among those common causes and does not establish what
the unpublished paper evaluator would produce. The queued paired greedy AMC
controls for SHS and whole-layer baseline remain necessary to test whether the
modal decoding paths also agree.
