# Qwen3-1.7B GPQA-Diamond-Freeform-126 Major-Checkpoint Eval Plan

Date: 2026-07-18

Status: **OWNER APPROVED - implementation/preparation authorized; remote launch deferred while the current Other/OOD wave is active.**

## Implementation Readiness Update / 实现进度

2026-07-18 local implementation is **SOURCE READY**. The approved config, pinned-asset builder,
checkpoint exporter, six-replica generation/matcher worker, autonomous controller, human-readable monitor,
and focused tests now exist. Synthetic exact-cover tests verify `126 -> 6 x 21` generation sharding and
`1260 -> 6 x 210` matcher sharding; Python compile, AST parsing, `bash -n`, and `git diff --check` pass.

这不等于 remote experiment complete：gated dataset revision、Qwen3-4B matcher receipt、10 个真实 checkpoint
exports、vLLM dispatch receipts 以及 1,260 个 matcher decisions 仍需在 current Other/OOD wave 结束后于远端产生。
Launcher 会在旧 wave screen 存活、GPU busy、asset receipt 缺失或 disk guard 不满足时 fail closed，绝不抢占
当前 evaluator。Authoritative preparation record:
`docs/experiment_records/2026-07-18_qwen3-1p7b-gpqa-diamond-freeform126-major-checkpoint-eval-record.md`.

## Purpose / 科学问题

Current `GPQA-Diamond` 是 four-choice discriminative evaluation。项目负责人批准新增一个 independently
named generative protocol：模型只看到 question，不看到 choices，并由 reference-aware matcher 判断
response 是否覆盖 ground-truth answer。核心问题是 current MCQ trajectory signal 能否迁移到真正的
free-form science answer，而不是仅来自 option recognition、choice-only shortcut 或 answer-letter
format behavior。

本轮覆盖 completed GRPO trajectory 的全部 paired checkpoints：TriGLU 与 whole-layer baseline 的
global steps `158/196/226/256/294`，共 `10` cells。不得以 nearby checkpoint 替换，不包含 untuned
base，也不修改任何 checkpoint、optimizer、RNG 或 training cursor。

## Approved Identity

- Run ID:
  `qwen3_1p7b_gpqa_diamond_freeform126_greedy3072_qwen3_4b_match_6x5090_steps158_196_226_256_294_20260718_v1`
- Screen: `qwen_gpqa_free126_majorsteps6_20260718_v1`
- Result root:
  `runs/freeform_eval/qwen3_1p7b_gpqa_diamond_freeform126_greedy3072_qwen3_4b_match_6x5090_steps158_196_226_256_294_20260718_v1/`
- Config:
  `configs/eval/qwen3_1p7b_gpqa_diamond_freeform126_majorsteps_6x5090_20260718_v1.yaml`
- Launcher:
  `scripts/run_qwen3_1p7b_gpqa_diamond_freeform126_majorsteps_6x5090_20260718_v1.sh`
- Monitor:
  `scripts/monitor_qwen3_1p7b_gpqa_diamond_freeform126_majorsteps_6x5090_20260718_v1.sh`
- Record:
  `docs/experiment_records/2026-07-18_qwen3-1p7b-gpqa-diamond-freeform126-major-checkpoint-eval-record.md`

## Immutable Cell Ledger

| Order | Cell | Global step | Action |
|---:|---|---:|---|
| 1 | TriGLU-158 | 158 | generate |
| 2 | baseline-158 | 158 | generate |
| 3 | TriGLU-196 | 196 | generate |
| 4 | baseline-196 | 196 | generate |
| 5 | TriGLU-226 | 226 | generate |
| 6 | baseline-226 | 226 | generate |
| 7 | TriGLU-256 | 256 | generate |
| 8 | baseline-256 | 256 | generate |
| 9 | TriGLU-294 | 294 | generate |
| 10 | baseline-294 | 294 | generate |

Every cell consumes the exact same ordered 126-row ledger。Rows are sorted by canonical `question_id`
and assigned by `ordered_index % 6`, producing exactly `21` rows per GPU with no residual、duplicate or
architecture-specific order drift。

## Dataset And Asset Contract

- Dataset repository: `nikhilchandak/freeform-datasets`;
- split/config: human-curated `gpqa_diamond` free-form subset;
- expected rows: exactly `126`;
- required fields: `question_id`, `question`, `answer`;
- choices must not appear in the generation prompt;
- access is gated. Authentication may come only from `HF_TOKEN` or an existing Hugging Face login;
  credentials must never enter config、manifest、logs or Git。

Operational amendment (2026-07-18): AutoDL reboot 后确认 remote 没有 HF credential。Asset builder therefore
prefers the gated canonical repo but may pin the same maintainer's public
`nikhilchandak/GPQA-diamond-free/gpqa-diamond-freeform-filtered.csv` when gated access fails. The fallback
must still pass exact 126-row/schema/unique-ID gates, and its repo revision/file provenance is written into the
manifest. We do **not** claim byte identity with the inaccessible gated artifact; later authenticated comparison
remains a provenance audit, not a reason to block the already-authorized scientific run.

The asset-preparation command must resolve and freeze the upstream commit SHA, save the raw normalized
126-row JSONL, produce a sorted ledger、field/identity audit、file SHA-256 and manifest, and fail closed on
missing/duplicate IDs、wrong row count、empty question/answer or schema drift。Production launch refuses to
run without this pinned manifest。

Matcher model is `Qwen/Qwen3-4B` in no-thinking mode。Its local model path、resolved upstream revision、
config/tokenizer file hashes and dtype must be frozen in a separate matcher asset receipt before launch。
Downloading either gated data or matcher weights is explicitly deferred until it cannot interfere with the
currently active remote evaluation wave。

## Generation Protocol

Primary generation deliberately matches the current longitudinal Other/OOD protocol rather than the 2025
answer-matching paper's sampling settings：

- backend: vLLM V1;
- six independent `TP=1` replicas, one per RTX 5090;
- seed `20260707`;
- greedy pass@1 (`temperature=0`, `do_sample=false`);
- response cap `3072`;
- `gpu_memory_utilization=0.85`, `max_num_seqs=128`, `max_num_batched_tokens=32768`;
- question-only prompt，允许 reasoning，但要求 final answer 位于 `<answer>...</answer>` tags；
- raw prompt、raw response、parsed tag、generated tokens、finish reason、cap hit、timing and dispatch
  receipt are retained per row。

The paper used temperature `0.6` and cap `16384` for thinking-model generation。This approved primary
protocol intentionally does not claim paper-faithful free-form reproduction; `greedy3072` is encoded in the
Run ID so a future paper-protocol cell cannot silently reuse this identity。

## Matching Protocol

Generation and matching are separate phases。All 1,260 candidate responses are generated first; then six
independent Qwen3-4B matcher replicas load once and score deterministic 210-row shards (`1260 / 6 = 210`)
to avoid ten repeated matcher loads。

- matcher: pinned `Qwen/Qwen3-4B`, `enable_thinking=false`, temperature `0`;
- prompt: the paper's published reference-aware ground-truth matching contract;
- input: question、reference answer、full candidate response; no incorrect choices;
- output schema: exactly one `<answer>0</answer>` or `<answer>1</answer>`;
- malformed output or missing decision is `matcher_failure`, never silently zero/correct;
- matcher only checks semantic/functional equivalence to the reference and must not independently solve or
  judge correctness without the reference;
- numeric responses follow the published `<1%` relative-error guidance where applicable。

Exact normalized string equality may be reported as a deterministic auxiliary diagnostic, but it never
overrides the primary matcher decision。The pipeline also emits a deterministic human-audit queue containing
all matcher failures plus seeded correct/incorrect examples and same-step architecture disagreements；human
review is a separately recorded follow-up, not silently fabricated by automation。

## Autonomous Execution And Resume

1. Fail closed if the active Other/OOD screen、another free-form controller or non-idle GPUs are detected；
2. verify pinned data/matcher manifests and all ten source checkpoints；
3. serially export one model cell, run six balanced generation shards, exact-merge, then remove only that
   temporary export after durable completion；
4. concatenate all ten complete generation cells into a canonical 1,260-row matcher ledger；
5. load six matcher replicas once, score six deterministic 210-row shards, exact-merge；
6. produce per-cell summaries、paired step table、failure-mode tables、audit queue and closeout receipt。

Every generation shard、cell merge、matcher shard、matcher merge and summary has an atomic completion marker。
Restart skips exact completed work and resumes only the first incomplete unit。Historical partial attempts and
errors remain provenance；a successful restart clears only current failure state。

## Human-Readable Monitor

One bounded monitor invocation displays only owner-relevant information：

- current phase (`WAITING/PREFLIGHT/EXPORT/GENERATE/MATCH/SUMMARIZE/COMPLETE/FAILED`);
- current cell、completed cells、generation rows and matcher rows;
- elapsed、recent speed、ETA;
- paired table by global step with TriGLU and baseline `correct/126`、accuracy、score delta;
- missing answer tags、cap hits、matcher failures and pending audit count;
- GPU utilization/memory、disk free and recent actionable error only。

No cross-benchmark hard average is created。Official MCQ `GPQA-Diamond` and this
`GPQA-Diamond-Freeform-126` score remain separate columns in later consolidated reporting。

## Acceptance Gates

- dataset manifest is pinned, `126/126` unique and exact-cover;
- each of ten generation cells is `126/126`, each GPU exactly `21`, zero duplicate/missing IDs;
- all cells use identical ordered ledger and generation settings;
- matcher ledger is exactly `1260/1260`, each GPU exactly `210`;
- every matcher decision is `0/1` or explicitly classified `matcher_failure`;
- per-cell denominator accounting closes exactly to 126;
- temporary exports are pruned only after durable cell completion；source checkpoints remain untouched;
- focused local and remote syntax/unit tests pass；compact source/docs evidence is committed and pushed;
- no launch occurs while the current Other/OOD wave is active。

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** deliberately deferred。This new all-vLLM generative protocol does not
  close HF-vs-vLLM parity or replace the official MCQ evaluator。
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deliberately deferred。Historical checkpoint dtype paths remain
  unchanged。
- **PENDING-03 Registered SHS CausalLM Route:** deliberately deferred。SHS is outside this TriGLU-baseline
  grid。
