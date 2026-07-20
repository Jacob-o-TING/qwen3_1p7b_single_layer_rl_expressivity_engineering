# AutoDL Poweroff Resume And GPQA-Freeform Handoff Record

Date: 2026-07-18

Status: **OLD WAVE RESUMED; GPQA HANDOFF ARMED**

## Incident / 事故

AutoDL balance exhaustion powered off the six-RTX-5090 instance during
`triglu_step256 / primary_humanevalplus`. Storage survived intact, but the old `screen` socket became dead and
all GPU processes disappeared. No evidence indicates checkpoint, source response, or completed shard corruption.

## Durable Resume Evidence

At reboot, five fresh Other/OOD cells were `CELL_COMPLETE`; `triglu_step256` had completed all seven staged
benchmarks plus LiveCodeBench and had four of six corrected HumanEval+ workers marked `WORKER_COMPLETE`.
Ranks `1/3/4/5` therefore remain reusable; only ranks `0/2` resume. Disk free was 121 GiB and all six GPUs were
0 MiB before launch.

The dead socket was wiped and `qwen_other_majorsteps6_20260718_v1` relaunched around the existing controller.
The resumed monitor reported `workers=4/6`; GPU 0 and 2 were active, confirming marker-aware continuation rather
than a full-cell rerun.

第一次 resume 随后暴露出一个 bookkeeping bug：unfinished TriGLU workers 会向上一次 failed attempt 留下的
`triglu_dispatch_receipts.jsonl` 继续 append，因此 validator 看见两条都合法但属于不同 runtime attempts 的
receipt，并以 `expected 1 dispatch receipts, found 2` fail closed。模型 generation、GPU runtime 与四个 completed
shards 都没有失败；controller 正是因为 receipt cardinality gate 返回 `rc=1` 才退出并释放 GPU。

Durable repair 会在 incomplete retry 前原子归档旧 receipt 为递增的 `attempt-NN` evidence，再为当前 attempt
创建 fresh receipt。Completed workers 仍由 `WORKER_COMPLETE` skip，不会被重跑。Remote `py_compile` 与 focused
`test_generator_backend.py` 共 7 tests passed；recovery check 再次观察到 screen online、state 回到
`triglu_step256 / primary_humanevalplus`，且只有 GPU 0/2 active at 25,535 MiB，符合仅续两个 unfinished shards
的 contract。

## Autonomous Serial Handoff

Old-wave controller 与 independent lightweight handoff watcher 共同执行这个 durable sequence：

1. resume the existing Other/OOD controller to `WAVE_COMPLETE`;
2. wait for local source deployment marker `runs/freeform_eval/GPQA_FREEFORM_HANDOFF_READY`;
3. launch `qwen_gpqa_free126_majorsteps6_20260718_v1` only after the old wave releases all GPUs;
4. prepare pinned GPQA dataset and Qwen3-4B matcher assets if absent;
5. run the approved GPQA-Diamond-Freeform-126 controller.

This avoids continuous external polling and remains autonomous if the owner's network disconnects again.

## Dataset Access Amendment

Remote HF credentials were absent and the preferred `nikhilchandak/freeform-datasets` repository is gated.
The asset builder now has a fail-closed fallback to the same maintainer's public
`nikhilchandak/GPQA-diamond-free/gpqa-diamond-freeform-filtered.csv`. Public schema is
`Question / Answer / Record ID / Canary String`; it maps to `question / answer / question_id`, then must pass
the unchanged 126-row, uniqueness, non-empty and six-way balance gates. Manifest records the exact public
revision and explicitly declines any unverified byte-identity claim against the gated repository.

Direct download preflight observed `126` rows, `126` unique `Record ID` values, zero empty `Question`, and zero
empty `Answer`; therefore the fallback satisfies the structural gate before remote handoff. Runtime preparation
will still resolve an immutable commit SHA and repeat all validation before writing its receipt.

`Qwen/Qwen3-4B` remains the approved no-thinking matcher. It is public, will be revision-pinned and downloaded
only after the old wave completes, preserving GPU-eval I/O priority.
