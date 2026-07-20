# Qwen3 Step-294 OOD TriGLU-Baseline-Base Plan

Date: 2026-07-16

Status: **APPROVED AND ACTIVE - durable handoff repair applied 2026-07-17**

## Identity

- Run ID: `qwen3_1p7b_ood_6x5090_step294_20260716_v1`
- Screen: `qwen_ood_step294_20260716_v1`
- Output root:
  `runs/ood_eval/qwen3_1p7b_ood_6x5090_step294_20260716_v1`
- Strict process order: TriGLU step-294 Math evaluation, TriGLU step-294 OOD,
  untuned Qwen3-1.7B-Base OOD, baseline third 98 updates, baseline step-294
  Math evaluation, then baseline step-294 OOD.
- Models run serially. Each model uses six independent TP=1 vLLM replicas.

The owner replaced the original post-`WAVE_COMPLETE` model order on 2026-07-16.
The GRPO controller itself now owns a durable pre-baseline barrier. Once
TriGLU's step-294 Math evaluation and retained export are durable, it writes
`TRIGLU_294_PRE_BASELINE_OOD_READY` and waits for the OOD runner to write
`PRE_BASELINE_OOD_COMPLETE`. The OOD watcher waits for that readiness marker,
waits for the evaluator and GPUs to become idle, then runs TriGLU OOD followed
by untuned-base OOD. Baseline's third 98 updates cannot start while the marker
is absent. After baseline's step-294 Math evaluation and export are durable and
the GRPO wave closes, the watcher runs baseline OOD. Network loss at the client
does not affect either screen.

The prior `SIGSTOP` interposer is retained only as historical evidence. On its
first live handoff it paused one controller-shell PID while a descendant raced
into baseline startup. OOD and baseline therefore briefly contended for the
same six GPUs; baseline vLLM failed its free-memory gate before update 197 and
the OOD runner also exposed evaluator preflight defects. The step-196 baseline
checkpoint, optimizer state, RNG state, and data cursor were not advanced.

## Benchmark Contract

The benchmark set follows the Qwen3 OOD suite in the paper:

- Code: HumanEval+, MBPP, LiveCodeBench;
- Reasoning: GPQA-Diamond, MMLU-Pro;
- Language: C-Eval, IFEval, MGSM.

GPU allocation is fixed as follows: GPU 0 receives HumanEval+ and MBPP, GPU 1
receives LiveCodeBench, GPU 2 receives GPQA-Diamond, GPU 3 receives MMLU-Pro,
GPU 4 receives C-Eval, and GPU 5 receives IFEval plus MGSM.

This is an explicitly named vLLM greedy pass@1 protocol with seed `20260707`,
`temperature=0`, and `max_tokens=3072`. The paper names the benchmarks and
unweighted category aggregation but does not report enough decoding detail to
claim strict evaluator parity. Results must therefore not be relabeled as the
paper's exact protocol.

Each category score is the unweighted mean of its benchmark scores. The report
also preserves an unweighted eight-benchmark OOD mean and an unweighted mean
of the three OOD category means. A later report may combine these with the
existing MathAvg to produce the paper-style four-category overall score, but
only while keeping protocol differences explicit.

## Code Execution Isolation

Code generation remains in the root-owned vLLM process, but generated Python
is reviewed by a separate privilege-dropped process. The review subprocess:

- runs as UID/GID 65534 with cleared groups and `no_new_privs`;
- receives an isolated scratch directory and minimal environment;
- has CPU, address-space, file-size, process-count, file-descriptor, and core
  limits;
- installs a seccomp filter after Python startup that denies network socket
  creation/use, process execution, namespace/mount operations, ptrace, BPF,
  and related privileged syscalls;
- captures bounded stdout/stderr and is killed after the benchmark timeout.

The container lacks Docker, Podman, usable namespaces, iptables/nftables, and
a system Python outside `/root`. The controller therefore temporarily changes
`/root` from mode 700 to traverse-only mode 701 so the dropped subprocess can
execute the pinned Python interpreter, and restores the exact original mode on
normal exit, failure, SIGINT, or SIGTERM. Failure to establish this isolation
is fail-closed for code ranks.

## Resume And Storage

Every GPU rank has an independent completion marker. On restart, finished
ranks are skipped and incomplete ranks use their latest EvalScope timestamp as
`use_cache`; no completed model or benchmark is deleted. The OOD controller
requires at least 100 GB free before launch and reuses existing step-294 merged
exports and the untuned base model without copying weights.

The split OOD runner accepts an explicit model subset. Completion of TriGLU
and untuned-base OOD writes `PRE_BASELINE_OOD_COMPLETE`; overall
`OOD_COMPLETE` is written only after baseline OOD also completes. A file lock
serializes retries, so an interrupted interposer can resume without duplicate
model execution.

Generic EvalScope datasets that do not provide the paper benchmark metadata
derive a deterministic request identity from the dataset namespace and the
rendered-prompt SHA-256. The paper benchmarks keep their explicit benchmark,
item, and sample identities unchanged. The local privilege-dropped sandbox
adapter explicitly bypasses EvalScope's unused `ms_enclave` import gate, while
IFEval's used `langdetect==1.0.9` dependency is pinned and checked before six
GPU engines are launched.

At the 2026-07-16 storage audit, `/root/autodl-tmp` had 330.27 GB available.
Two remaining 98-step stages, their retention schedule, exports, math evals,
and a 10-20 GB OOD reserve were conservatively projected to consume 212-222
GB at peak, leaving approximately 108-118 GB. The capacity gate therefore
passed, with checkpoints rather than OOD traces as the dominant storage cost.

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** deferred. This OOD protocol is a new
  consistent all-model vLLM comparison, not closure of the HF/vLLM matrix.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deferred. The historical mixed-
  precision TriGLU checkpoint remains the evaluated artifact.
- **PENDING-03 Registered SHS CausalLM Route:** deferred. SHS is not a model in
  this OOD wave.

## 2026-07-17 LiveCodeBench Six-GPU Amendment

The owner replaced the original one-GPU LiveCodeBench allocation after the
first attempt spent 38 minutes preparing overlapping EvalScope subsets and
then failed before generation because `/root` was full. That attempt generated
and reviewed zero LiveCodeBench samples. The ModelScope cache was subsequently
migrated byte-exactly to `/root/autodl-tmp/cache/modelscope`; `/root/.cache/modelscope`
is now a compatibility symlink, preserving all downloaded data while freeing
the system filesystem.

The amended LiveCodeBench protocol is
`livecodebench_release_latest_6way_hashshard_20260717_v1`:

- freeze the canonical EvalScope `release_latest` subset at 1,055 samples;
- derive a SHA-256 identity from stable source fields and assign each sample by
  `int(identity[:16], 16) % 6`;
- run six independent TP=1 vLLM replicas on disjoint shards, retaining greedy
  decoding, seed `20260707`, response cap `3072`, and the privilege-dropped
  local sandbox;
- require 1,055 unique identities, zero duplicates, exact receipt/report
  counts, and deterministic weighted aggregation before writing completion;
- preserve every shard's predictions, reviews, reports, logs, and receipts.

The non-LiveCodeBench OOD cells remain a first parallel stage. LiveCodeBench
then receives all six GPUs as a second stage. Existing completed cells are
resumed by marker and are not regenerated. The same staged protocol applies to
TriGLU, untuned base, and baseline so architecture comparisons retain one
evaluation contract.

The EvalScope default metadata lists 28 overlapping release and cumulative
subsets. Splitting those subset names across GPUs would duplicate benchmark
questions and make a merged score ill-defined. The amended protocol therefore
shards the single canonical `release_latest` sample set rather than sharding
the overlapping subset list.

## Approved 2026-07-17 Baseline Step-196 Control Amendment

The owner approved an additional baseline step-196 OOD control because the
untuned-base scores differ from the paper anchor and may indicate protocol
drift. Its separate identity is
`qwen3_1p7b_ood_6x5090_baseline_step196_20260717_v1`, model label
`baseline_step196`, screen `qwen_baseline_step196_ood6_20260717_v1`, and future
record `2026-07-17_baseline-step196-ood-control-record.md`. It evaluates the
durable `baseline_step_196` export with the same seed, greedy decoding, 3,072
token cap, benchmark revisions, vLLM profile, and corrected evaluator contracts.

The execution order is amended to TriGLU step-294 OOD, untuned-base OOD,
baseline step-196 OOD, then release the existing pre-baseline barrier so
baseline updates 197-294 can resume. A directory-valued barrier interposer is
used while the legacy untuned-base process finishes: the legacy `touch`
succeeds, but the GRPO controller's `-f` completion check remains false. The
interposer is replaced by a regular marker only after the step-196 OOD control
passes its completion gates.

The old OOD scheduler assigned different benchmarks to individual GPUs and
therefore left five GPUs idle behind the long MMLU-Pro straggler. The step-196
control introduces benchmark-stage serial scheduling: each benchmark is
deterministically hash-sharded across all six GPUs, merged fail-closed, and only
then advances to the next benchmark. Test-sample sharding occurs after common
few-shot prompt construction, so all ranks retain identical demonstrations.
Every stage requires exact identity coverage, zero duplicates, six completion
markers, and a weighted merge of the six reports. LiveCodeBench retains its
existing source-record-specific six-way sharder.

The already 65%-complete untuned-base MMLU-Pro legacy cell is not restarted;
discarding that work would cost more time than it saves. The staged-six layout
applies to the new baseline step-196 control and is the preferred topology for
subsequent full OOD comparisons.

### 2026-07-17 Staged-Sharding Recovery Amendment

The first baseline-step196 GPQA stage failed closed at `188/198` identities.
The six workers and model generations completed, but ownership had been derived
from a hash containing rank-local formatted payload. Because worker bootstrap
can perturb formatting RNG, this is not a stable ownership key across six
independent processes.

Ownership is now derived from `(dataset, subset, canonical sample_index)` only.
The formatted payload remains covered by a separate SHA-256 in every receipt.
The failed attempt is retained, and the resumed stage passed exact `198/198`
coverage with score `22.727%`. No macro scientific setting changed.

Both human-readable monitors must expose the inserted baseline-step196 control.
For every staged benchmark they report six-shard progress, partial weighted
score and sample count while incomplete, and final score with exact identity
coverage after merge. A partial score is never promoted to a final benchmark
result.

### 2026-07-17 HumanEvalPlus Protocol-Diagnostic Amendment

The baseline step-196 control completed all eight stages. Its HumanEval+ score
is `51/164 = 31.097%`; the complete control results are preserved in
`docs/experiment_records/2026-07-17_baseline-step196-ood-control-record.md`.

The untuned-base HumanEval+ score remains `1/164 = 0.610%` after parser-v2 and
fixed-sandbox review, while untuned-base LiveCodeBench is `7.396%`, matching the
paper's `7.4%` anchor to rounding. The current generator applies the Qwen chat
template to EvalScope's zero-shot HumanEval+ instruction for every model. This
is retained as the internally consistent chat protocol, but it is not treated
as a paper-comparable base-model capability estimate.

A separately named prompt-protocol matrix is documented in
`docs/experiment_plans/2026-07-17_qwen3-1p7b-humanevalplus-prompt-protocol-matrix-plan.md`.
The owner approved it, and its 32-task canary isolated the dominant mismatch to
the chat template. The full corrected untuned-base run then scored
`97/164 = 59.146%` with zero collapse loops under the raw EvalScope instruction
without chat wrapping. This establishes material recovery but not exact paper
parity because the result exceeds the paper's 44.5% anchor and the paper does
not disclose the full evaluator contract.

All historical chat-protocol OOD results remain immutable. Full-164 corrected
reruns for TriGLU step-294 and baseline step-196 use the exact untuned-base task
ledger, seed, response cap, parser, sandbox, and six-way shard protocol. Both
human-readable monitors expose the old and corrected HumanEval+ scores plus
partial corrected-protocol progress while these reruns are active.

The corrected reruns completed with untuned base at `97/164 = 59.146%`, TriGLU
step-294 at `100/164 = 60.976%`, and baseline step-196 at
`100/164 = 60.976%`. Their corrected equal-weight CodeAvg values are 29.714%,
33.439%, and 32.980% respectively. The OOD summarizer now reports legacy
chat/parser and prompt-corrected HumanEval+/CodeAvg side by side. It does not
overwrite the original OOD benchmark/category means with a mixed protocol.

Once all four pre-baseline evidence groups were durable (TriGLU OOD,
untuned-base OOD, baseline step-196 OOD, and the three corrected HumanEval+
cells), the empty directory interposer was atomically replaced by a regular
`PRE_BASELINE_OOD_COMPLETE` marker. The unchanged GRPO controller released its
barrier and resumed baseline step 196 toward target 226 using the existing
checkpoint, optimizer state, RNG state, data cursor, step-196 reference, and
approved cosine schedule. This is the planned continuation, not a fresh run.
