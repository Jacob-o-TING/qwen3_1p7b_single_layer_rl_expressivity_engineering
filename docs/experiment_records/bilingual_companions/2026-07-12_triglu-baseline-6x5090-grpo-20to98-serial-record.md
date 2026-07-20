# TriGLU + Baseline 6x5090 GRPO 20-to-98 Serial Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-12_triglu-baseline-6x5090-grpo-20to98-serial-record.md](../2026-07-12_triglu-baseline-6x5090-grpo-20to98-serial-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **TriGLU + Baseline 6x5090 GRPO 20-to-98 Serial Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


Status: ACTIVE

Approved order: TriGLU-20, eval, baseline-20, eval, TriGLU-98, eval, baseline-98, eval.

The run starts from the same untuned Qwen3-1.7B-Base revision for both variants. Runtime,
dataset hashes, topology, canary receipts, launch timestamp, measured ETA, milestones, and
evaluation summaries are appended as the autonomous wave progresses.

Owner-approved final decision before launch: use all six ranks with `504/126/6` batch
arithmetic. This is an explicit 1.56% exposure deviation from paper `512/128/8`, accepted
to avoid leaving two of six GPUs idle. Both variants remain exactly matched. The temporary
four-rank exact-paper alternative was never launched.

Prelaunch incident: the first controller attempt correctly stopped before model loading
because it was pointed at the canonical analysis parquet (`problem`, `solution`, `answer`,
`messages_json`) rather than its already-materialized veRL view. veRL defaulted to a
missing `prompt` field and produced a zero-row dataset. No parameters or checkpoints were
created. The production configs now explicitly bind both variants to the existing
`numina_math_cot_50k_decontam_v3_verl` train/val files, whose row order derives from the
same v3 materialization; both file hashes are fail-fast pinned.

Prelaunch integration incident: the next canary reached rollout weight synchronization but
stopped at update 0 because vLLM had been initialized from the vanilla Base path while the
actor exposed TriGLU wrapper parameter names. No optimizer update or checkpoint was made.
Production now initializes TriGLU actor and vLLM from the already validated 2x5090
exact-noop TriGLU export; baseline continues to initialize from the same underlying untuned
Base revision. The controller fail-fast checks the custom `qwen3_triglu` config before work.

Live research note from the owner: investigate whether policy-level KL unnecessarily limits
the new TriGLU third branch. Ordinary GRPO KL is defined on actor/reference output
distributions, not directly on a layer or branch, so gradients reach the third branch only
through its effect on logits. Conventional LoRA SFT generally has no such policy KL, whereas
LoRA RLHF/GRPO often does. This observation is recorded for a separately named post-wave
regularization ablation; the active matched run remains unchanged.

Step-20 eval incident: the initial FSDP merge stopped because the TriGLU Hugging Face
checkpoint requires custom code and `verl.model_merger` defaulted to
`trust_remote_code=False`. Bash command substitution also captured merger stdout as if it
were the model path and permitted a false empty `EXPORT_COMPLETE` marker. No evaluation
rows were generated and no training state was changed. Recovery hardens the controller by
passing `--trust-remote-code`, routing merger diagnostics outside command substitution,
validating config plus safetensor weights before the marker, rejecting export failures, and
skipping already-complete durable training segments on restart.

The first hardened retry exposed a second packaging defect: the checkpoint's copied
`triglu_hf_model.py` imports `TRIGLU_ARCHITECTURE` from a package `__init__` that veRL did
not include. Recovery now builds a staging-only actor directory: immutable FSDP shards are
symlinked, Hugging Face metadata is copied, the missing constant is inlined only in that
copy, and an isolated Transformers module cache is used. Original checkpoint model,
optimizer, RNG, and metadata files remain unchanged.

The staging retry then showed that veRL had also omitted the two files named by the
checkpoint `auto_map`: `configuration_qwen3_triglu.py` and
`modeling_qwen3_triglu.py`. The staging repair now creates the same tiny import wrappers
used by the validated deployment export and preflights AutoConfig plus both dynamic model
classes before invoking the expensive merger.

Step-20 milestones completed for both variants, including their six-GPU parallel evals.
The human-readable monitor originally folded AMC greedy pass@1 reviews into the sampled
AMC Avg@32 cell because the summarizer classified the `paper_amc23` filename before the
more specific `amc_greedy` protocol directory. This was a reporting-only defect; no
generation, review, training, checkpoint, or stored evaluation artifact was changed.
After correcting label precedence, the durable step-20 AMC results are:

- TriGLU: Avg@32 `280/1280 = 21.88%`; greedy pass@1 `15/40 = 37.50%`.
- Baseline: Avg@32 `295/1279 = 23.06%`; greedy pass@1 `15/40 = 37.50%`.

The baseline sampled denominator remains an honest `1279`: all 1280 sampled generations
exist, but one row has no score-bearing review. The monitor now names the two cells
`paper_amc23_avg_at_32` and `paper_amc23_greedy_pass_at_1` while preserving the underlying
summary JSON keys for compatibility. The autonomous wave continued into the TriGLU
step-20-to-98 segment without interruption.

Pending obligations carried forward: PENDING-01 Eval Parity Matrix, PENDING-02 pure-BF16
architectures, and PENDING-03 registered SHS CausalLM are all deliberately deferred.

## 2026-07-14 Step-98 And Metric-Hierarchy Snapshot

At the bounded 2026-07-14 check, TriGLU step 98 and its six-GPU evaluation were
complete. Baseline training was healthy at step 75/98 with the step-75
checkpoint present; baseline step-98 results remain pending and are not
inferred here.

TriGLU step-98 evaluation:

| Benchmark | Result |
|---|---:|
| AMC Average@32 | 410/1280 = 32.03% |
| AMC greedy pass@1 | 15/40 = 37.50% |
| GSM8K | 1091/1319 = 82.71% |
| MATH-500 | 319/500 = 63.80% |
| OlympiadBench | 181/675 = 26.81% |
| Equal-weight MathAvg, with AMC Average@32 | 51.3401 |
| Whole-50K weighted proxy, with AMC greedy | 64.0703 |

For comparison, step-20 MathAvg was `48.6771` for baseline and `48.7262` for
TriGLU. TriGLU therefore gained `+2.6139` MathAvg points between steps 20 and
98, while AMC greedy remained `15/40` for baseline-20, TriGLU-20, and
TriGLU-98. The reporting hierarchy is now explicit: MathAvg with AMC Average@32
is primary; the training-mix weighted proxy with AMC greedy is secondary; AMC
greedy alone is diagnostic.

Normal six-GPU vLLM evaluation phases have measured approximately five to nine
minutes per checkpoint after export. The owner therefore approved retrospective
evaluation of existing step-30 and step-60 checkpoints for both variants after
the active wave completes, producing a `20/30/60/98` learning curve. If that
curve remains positive, the same checkpoints may continue to cumulative
`128/158/196`; a new architecture starts from the untuned base and is evaluated
at `30/60/98`.

The current loader uses a deterministic shuffled 50,000-row permutation,
`batch_size=504`, and `drop_last=True`. Step 98 consumes 49,392 rows and leaves
608 rows unconsumed. This milestone is therefore a matched
`98-step near-one-pass`, not an exact full epoch. Both variants must share the
same omitted rows; their IDs, hash, and source composition must be preserved.
No irregular partial batch is authorized. Exact all-row carry-over remains a
separate future sampler protocol.

## 2026-07-14 Selective AMC Gain Research Note

TriGLU's step-20 to step-98 improvement is highly selective: AMC Average@32
rises `+10.1563` percentage points, while OlympiadBench rises `+1.3333`, GSM8K
falls `-0.8340`, MATH-500 falls `-0.2000`, and AMC greedy remains unchanged.
This pattern motivates two non-causal working hypotheses: sampled-group GRPO
may improve probability mass over rewarded AMC-like trajectories before the
greedy mode changes, and the version-1 training-mix proxy may underweight
K12/curriculum transfer into AMC by assigning foundation sources almost
exclusively to GSM8K. The full interpretation, limits, and discriminating tests
are recorded in
`2026-07-13_training-mix-weighted-eval-composite-record.md`.
