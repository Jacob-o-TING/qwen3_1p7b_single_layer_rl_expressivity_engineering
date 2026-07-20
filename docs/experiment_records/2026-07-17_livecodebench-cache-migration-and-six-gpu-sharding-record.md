# LiveCodeBench Cache Migration And Six-GPU Sharding Record

Date: 2026-07-17

Status: **COMPLETE after cached-prediction evaluator recovery.** Six-GPU
generation covered 1,055 unique `release_latest` rows. The initial aggregate
`0/1055` report is invalid; the corrected frozen-prediction review is
`107/1055 = 10.142%`.

## First Attempt Boundary

The first TriGLU LiveCodeBench process did not evaluate the benchmark. It
downloaded and prepared part of EvalScope's ModelScope dataset cache, reached
`Running[eval] 0/1`, and then failed while writing an Arrow dataset with
`OSError: [Errno 28] No space left on device`. It generated zero model answers,
reviewed zero programs, and produced no score or completion receipt. Later
execution is therefore the first formal LiveCodeBench evaluation, not a repeat
of evaluated samples.

The failure was caused by cache placement rather than data-disk capacity. The
30 GB system filesystem was 99% full while `/root/autodl-tmp` retained more
than 200 GB. ModelScope alone occupied approximately 18 GB under
`/root/.cache/modelscope`.

## Cache Migration

The ModelScope cache was migrated to:

```text
/root/autodl-tmp/cache/modelscope
```

The compatibility path `/root/.cache/modelscope` is now a symlink to that
data-disk directory. Before the swap, the source and destination matched at
1,007 regular files and 18,990,780,305 regular-file bytes, with zero rsync
dry-run delta. The post-swap cache state was readable. The old system-disk copy
was removed only after those gates passed. System-disk utilization fell from
99% to 33%, leaving approximately 21 GB free. The active MMLU-Pro evaluator
held no open ModelScope-cache files and completed normally.

## Parallel Protocol

Approved identity:
`livecodebench_release_latest_6way_hashshard_20260717_v1`.

The EvalScope metadata names 28 release and cumulative subsets that overlap.
Partitioning those names across GPUs would duplicate questions and make the
merged score ambiguous. This protocol instead freezes the canonical
`release_latest` subset and shards its samples.

Each source row receives a SHA-256 identity over stable source fields. Its rank
is `int(identity[:16], 16) % 6`. A zero-GPU cache preflight found:

- total rows: 1,055;
- unique identities: 1,055;
- duplicate identities: 0;
- shard sizes: 175, 172, 173, 155, 195, and 185.

Each GPU owns one independent TP=1 vLLM replica. The six shards preserve greedy
decoding, seed `20260707`, maximum response length 3,072 tokens, EvalScope's
LiveCodeBench adapter, and the project privilege-dropped local code sandbox.
Every shard preserves its predictions, reviews, report, generation receipts,
and dataset-assignment receipts.

Merge is fail-closed. It requires all six completion markers, exact
receipt/report count agreement, correct hash-to-rank assignment, 1,055 unique
identities, and zero duplicates before writing the aggregate summary or the
canonical `code_lcb/RANK_COMPLETE` marker.

## OOD Integration

For each model, the staged OOD runner first executes all non-LiveCodeBench
cells in parallel. It then assigns all six GPUs to LiveCodeBench. Already
completed TriGLU cells are marker-resumed and will not be regenerated. The same
staged protocol applies to untuned base and baseline, keeping the comparison
contract fixed across models.

Remote verification before launch passed:

- Python compile checks for the evaluator, sharder, and merger;
- Bash syntax checks for the parallel launcher, OOD runner, and monitor;
- focused LiveCodeBench sharding tests: 3/3;
- focused OOD queue tests: 4/4;
- canonical 1,055-row identity ledger assertions.

## Live Launch Receipt

The production launcher started at `2026-07-17T19:11:17+08:00` in detached
screen `qwen_triglu_lcb6_release_latest_20260717_v1`. Its controller log is:

```text
runs/ood_eval/qwen3_1p7b_ood_6x5090_step294_20260716_v1/logs/livecodebench_parallel6_resume_20260717_v1.log
```

All five completed TriGLU non-LiveCodeBench cells emitted
`OOD_RANK_ALREADY_COMPLETE` and were skipped: MMLU-Pro, HumanEval+/MBPP,
GPQA, C-Eval, and IFEval/MGSM. No completed benchmark was regenerated.

The controller then emitted one `LCB_SHARD_START` receipt for each shard
`00` through `05`, mapped one-to-one to GPUs `0` through `5`. A bounded live
check found all six GPUs decoding concurrently at 30-41% utilization and
approximately 28,831 MiB allocated per GPU. Both the LiveCodeBench screen and
the parent GRPO controller screen remained detached and alive. No traceback,
CUDA OOM, or disk-full error was observed at this launch gate.

The merge subsequently completed with 1,055 unique predictions, mean generated
length 907 tokens, median 674, p90 1,811, and 100 length-cap hits. The original
review nevertheless reported zero because the project sandbox backend returned
program stdout only as `stdout`, while EvalScope's LiveCodeBench adapter reads
the sandbox contract field named `output`. At least 147 historical failed-case
receipts literally contained `TEST_PASSED`, proving that the all-zero report was
an evaluator artifact.

The sandbox now retains `stdout` for diagnostics and also exposes the same text
as `output`. A focused regression test locks this contract. A first cached
rescore attempt was invalidated because it stringified EvalScope's persisted
OpenAI-style `model_output` object; that attempt is preserved under an
`invalid_attempt_1_model_output_dict_stringification` directory and contributes
no score. The corrected rescorer reads the canonical persisted assistant
message, verifies all six source-prediction SHA-256 values, and performs no
model loading or generation.

## Final Cached Review Result

| Shard | Passed | Rows | Score |
|---|---:|---:|---:|
| 00 | 19 | 175 | 10.857% |
| 01 | 17 | 172 | 9.884% |
| 02 | 17 | 173 | 9.827% |
| 03 | 15 | 155 | 9.677% |
| 04 | 14 | 195 | 7.179% |
| 05 | 25 | 185 | 13.514% |
| **Merged** | **107** | **1,055** | **10.142%** |

The review receipt SHA-256 is
`ced757612dca339de6c96531baf4d9b8c3053a30d5ebafa6a08ed52e293b8776`.
The source predictions are unchanged. This result repairs scoring only and does
not reinterpret generation quality, change prompts, or mutate the checkpoint.

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`.

- **PENDING-01 Eval Parity Matrix:** deferred. This vLLM OOD run does not close
  the HF-versus-vLLM parity matrix.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deferred. The evaluated TriGLU
  checkpoint retains its historical mixed-precision custom path.
- **PENDING-03 Registered SHS CausalLM Route:** deferred. SHS is not part of
  this OOD wave.
