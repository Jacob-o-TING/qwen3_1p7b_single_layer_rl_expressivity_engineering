# TriGLU-Baseline Major-Checkpoint Other Eval Plan / 主要检查点域外评估计划

Date: 2026-07-18

Status: **OWNER APPROVED - implementation and launch authorized**

## 目的与科学问题 / Purpose

这轮 experiment 要把 completed GRPO trajectory 从 global step `158` 到
`294` 的主要 checkpoints 做成 paired Other/OOD evaluation grid。核心问题不是只看
final checkpoint，而是观察 TriGLU 与 whole-layer baseline 的 out-of-domain
capability trajectory 是否随 training stage 稳定分叉，以及 Math gains 是否伴随
code、reasoning 或 multilingual capability 的 tradeoff。

## Approved Identity

- Run ID:
  `qwen3_1p7b_other_eval_6x5090_triglu_baseline_steps158_196_226_256_294_20260718_v1`
- Screen: `qwen_other_majorsteps6_20260718_v1`
- Result root:
  `runs/ood_eval/qwen3_1p7b_other_eval_6x5090_triglu_baseline_steps158_196_226_256_294_20260718_v1/`
- Config:
  `configs/eval/qwen3_1p7b_other_eval_majorsteps_6x5090_20260718_v1.yaml`
- Launcher:
  `scripts/run_qwen3_1p7b_other_eval_majorsteps_6x5090_20260718_v1.sh`
- Monitor:
  `scripts/monitor_qwen3_1p7b_other_eval_majorsteps_6x5090_20260718_v1.sh`
- Record:
  `docs/experiment_records/2026-07-18_triglu-baseline-major-checkpoint-other-eval-record.md`

## Immutable Cell Ledger

Approved global steps are exactly `158`, `196`, `226`, `256`, and `294` for both
architectures. No nearby checkpoint may silently replace one of these steps。

| Global step | TriGLU | Baseline | Execution action |
|---:|---|---|---|
| 158 | required | required | run both |
| 196 | required | required | run TriGLU; import completed baseline evidence |
| 226 | required | required | run both |
| 256 | required | required | run both |
| 294 | required | required | import completed TriGLU evidence; run baseline |

The two imported cells are immutable controls:

- `TriGLU-294` from
  `runs/ood_eval/qwen3_1p7b_ood_6x5090_step294_20260716_v1/triglu`;
- `baseline-196` from
  `runs/ood_eval/qwen3_1p7b_ood_6x5090_baseline_step196_20260717_v1`.

Import means link-and-verify with source receipts and hashes; it never means copy,
rescore, overwrite, or regenerate the completed predictions。

## Benchmark And Protocol Contract

每个 newly evaluated cell 必须完成相同的 eight-benchmark suite：

- Reasoning: `GPQA-Diamond`, `MMLU-Pro`;
- Code: `HumanEval+`, `MBPP`, `LiveCodeBench release_latest`;
- Language: `C-Eval`, `IFEval`, `MGSM`.

Generation 使用 six independent TP=1 vLLM replicas、seed `20260707`、greedy
pass@1、response cap `3072`。Generic benchmarks 采用 staged-six deterministic
sample sharding：每个 benchmark 先在六张 GPU 上 exact-cover and merge，满足 zero
duplicates、expected identities、six rank markers 后才进入下一 benchmark。
LiveCodeBench 保持现有 canonical `release_latest` 1,055-row six-way hash sharder。

`PRIMARY` code protocol 使用 raw no-chat HumanEval+、parser-v2、privilege-dropped
sandbox 与 corrected LiveCodeBench。为保留 historical comparability，每个新 cell
也保留 `HERITAGE` chat-protocol HumanEval+，但它不得混进 PRIMARY aggregate。

## Checkpoint And Export Contract

Step `158` 从 interleaved run 的 exact checkpoint 导出；steps
`196/226/256/294` 从 third-segment run 的 exact checkpoints 导出。Export 必须：

1. validate all six-rank actor checkpoint parts and global-step identity;
2. restore the exact architecture/config for TriGLU or baseline;
3. write to a cell-specific temporary export directory;
4. verify `config.json`, safetensors, architecture dispatch, and completion receipt;
5. delete only the temporary export after that cell's complete evaluation and
   durable summary, never delete a source checkpoint or evaluation evidence。

## Autonomous Order And Resume

执行顺序按 same-step comparison 优先：

1. TriGLU-158, baseline-158;
2. TriGLU-196, import baseline-196;
3. TriGLU-226, baseline-226;
4. TriGLU-256, baseline-256;
5. import TriGLU-294, baseline-294.

Every benchmark shard、merged stage、corrected HumanEval+ cell、model-cell summary
和 imported-evidence receipt 都有 independent completion marker。Restart 必须 skip
completed work and resume only the incomplete shard/stage；网络中断不能阻止 serial
queue 自主推进。

## Human-Readable Monitor Contract

Monitor 的最上方显示 current cell、benchmark/stage、completed cells、partial score、
elapsed time、recent speed 与 ETA。结果区按 global step `158/196/226/256/294`
排列，每个 step 同一行并列 `TriGLU` 与 `baseline`，不能按 architecture 分成两张难以
比较的表。

结果 presentation 保留现有 hierarchy：

1. `PRIMARY corrected-protocol` complete/partial table；
2. eight individual benchmark scores and Code/Reasoning/Language means；
3. separately labeled `HERITAGE chat-protocol` table；
4. source/import status、GPU utilization、disk free 和 recent actionable errors。

Partial results 必须显示 numerator/denominator 和 `partial` label，绝不能冒充 final。

### 2026-07-18 Owner Reporting Amendment

项目负责人明确撤销 `OOD-8` 与 `OOD-category` 两个 hard averages。跨 benchmark 或跨
category 的 equal-weight average 缺少足够的 scientific justification，容易让一个
scalar 掩盖 capability profile。Monitor、human-readable summary 与最终 report 不再
显示这两个 derived aggregates；八个 benchmark 原分数以及 Code、Reasoning、
Language 各 category 内部的 mean 继续保留。Historical JSON fields 不篡改，
generation、scoring 与 raw evidence 均不受影响。

## Next-Stage Free-Response Evaluation Extension

Status: **PLANNING AMENDMENT ONLY - no implementation, download, or launch authorization.**

当前 eight-benchmark suite 同时包含 four-choice `GPQA-Diamond` 与 up-to-ten-choice
`MMLU-Pro`，但它们仍可能测到 option recognition、choice-only shortcut 与 parser-format behavior。
下一阶段应增加不提供 answer choices 的 science evaluations，从而判断 architecture/RL gain 是否能够
迁移到 independently generated answers。

### Candidate A: GPQA-Diamond-Freeform-126

项目负责人决定第一优先级使用基于
[`nikhilchandak/freeform-datasets`](https://huggingface.co/datasets/nikhilchandak/freeform-datasets)
的 human-curated `126`-row GPQA-Diamond free-response subset。模型只看到 question，不看到 choices；
答案不保证是纯数字，因此需要 reference-aware answer matching。相比 `869`-row SciBench，它能以更低
generation cost 先回答最直接的问题：current GPQA MCQ signal 能否迁移到 generative answer。Launch
前必须另外冻结：

- exact 126-row identity ledger 以及与 official GPQA Record ID 的 mapping；
- candidate checkpoint grid 与 generation protocol；
- matcher model/revision、prompt、temperature、seed、decision schema 与 retry policy；
- matcher disagreement handling、human spot-audit sample 与 `uncertain` category；
- generation failure、matcher failure 与 semantically wrong answer 的独立统计。

该结果必须命名为 `GPQA-Diamond-Freeform-126`，不得覆盖 official four-choice
`GPQA-Diamond`。Current MCQ score、strict-final-answer diagnostic 与 free-response score 应并列保留，
用于分解 choice recognition、format compliance 与 genuine generative science capability。

### Candidate B: SciBench Numerical Free Response

第二阶段扩展使用 `SciBench`：约 `869` 道 college-level mathematics、physics、chemistry open-ended
problems，textbook evaluation 聚焦 single numerical final answers。该 cell family 应使用 separately
approved run identity，并在执行前冻结：

- exact dataset revision、split、row ledger 与 SHA-256；
- prompt、reasoning/final-answer format、seed、response cap 与 stop policy；
- numeric parser 对 integer、decimal、fraction、scientific notation 与 equivalent forms 的规则；
- absolute/relative tolerance、unit handling、missing/ambiguous answer 与 cap-hit policy；
- sample-level raw response、extracted value、reference value、score 与 failure-mode artifact；
- six-GPU topology-neutral deterministic sharding、partial merge 与 resumable completion markers。

SciBench score 必须保持独立；不得并入已经 retired 的 `OOD-8` 或 `OOD-category` hard averages。
它的主要 scientific role 是区分 option recognition 与 actual quantitative derivation。

### Proposed Decision Order

1. 先完成 current major-checkpoint Other/OOD wave 与 closeout；
2. 项目负责人另行批准 exact run/config/screen/result/report naming 与 checkpoint cells；
3. 先做 small deterministic preflight，验证 numeric parser 或 answer matcher；
4. `GPQA-Diamond-Freeform-126` 先行，利用 `126` rows 做 low-cost generative-science diagnostic，
   并把 matcher uncertainty 作为 first-class evidence；
5. 再运行 `SciBench`，把 positive/ambiguous GPQA free-form finding 扩展到更大规模的 deterministic
   numerical-science evaluation；
6. 保留 `GPQA-Diamond`、`MMLU-Pro`、`SciBench` 与 free-form scores 的 separate columns，不制造新的
   unjustified cross-benchmark scalar。

这项 next-stage extension 不完成或替代 Mandatory PENDING Registry 中的任何 gate；特别是它不能被
误记为 Eval Parity Matrix closeout。

## Storage And Safety

Launch-time remote free space is `127 GB`。Existing complete Other-eval output is
approximately `0.4-0.6 GB` per model cell；eight new cells are expected to add roughly
`3-6 GB` plus one serial temporary export。Source checkpoints remain the dominant
storage consumer。A conservative `40 GB` free-space guard applies before each new
cell；temporary exports are serialized and removed only after durable completion。

No training state、optimizer、RNG、data cursor、source checkpoint、existing prediction,
or historical report may be mutated by this evaluation wave。

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** deferred. This wave expands the consistent
  all-model vLLM protocol across checkpoints but does not close HF-vs-vLLM parity。
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deferred. Historical mixed-precision
  TriGLU checkpoints are evaluated without architecture conversion。
- **PENDING-03 Registered SHS CausalLM Route:** deferred. SHS is outside this
  two-architecture checkpoint grid。
