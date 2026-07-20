# TriGLU-Baseline Major-Checkpoint Other Eval Record / 主要检查点域外评估记录

Date: 2026-07-18

Status: **ACTIVE - autonomous evaluation wave launched and healthy; final scientific closeout pending.**

## Purpose / 科学问题

本轮把 completed GRPO trajectory 的 global steps `158/196/226/256/294`
组成 paired TriGLU-versus-baseline Other/OOD grid。目标不是只看 final checkpoint，
而是观察 code、reasoning、multilingual 与 instruction-following capability 随训练阶段的
trajectory，尤其检查 Math gain 是否伴随可重复的 domain tradeoff。

本记录是 running experiment 的 authoritative bilingual record。`ACTIVE` 只表示 launcher、
resume contract、monitor 与第一项 fresh evaluation 已经真实运行；在全部十个 cells exact-merge、
compact evidence 回收到本地并完成 closeout 之前，不得把本轮写成 `COMPLETE`。

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
- Hardware: one host, `6 x RTX 5090`, six independent `TP=1` vLLM V1 replicas.

## Immutable Cell Ledger

| Order | Cell | Action |
|---:|---|---|
| 1 | TriGLU-158 | fresh evaluation |
| 2 | baseline-158 | fresh evaluation |
| 3 | TriGLU-196 | fresh evaluation |
| 4 | baseline-196 | import and verify completed evidence |
| 5 | TriGLU-226 | fresh evaluation |
| 6 | baseline-226 | fresh evaluation |
| 7 | TriGLU-256 | fresh evaluation |
| 8 | baseline-256 | fresh evaluation |
| 9 | TriGLU-294 | import and verify completed evidence |
| 10 | baseline-294 | fresh evaluation |

因此实际 workload 是 `8 fresh + 2 imported` cells。Import 只允许 link、hash/receipt
verification 与 presentation reuse；不得 regenerate、rescore 或 overwrite 已完成 predictions。
两个 immutable sources 分别是：

- TriGLU-294:
  `runs/ood_eval/qwen3_1p7b_ood_6x5090_step294_20260716_v1/triglu`;
- baseline-196:
  `runs/ood_eval/qwen3_1p7b_ood_6x5090_baseline_step196_20260717_v1`.

## Protocol Contract

每个 fresh cell 使用同一个 eight-benchmark suite：`GPQA-Diamond`、`MMLU-Pro`、
`HumanEval+`、`MBPP`、`C-Eval`、`IFEval`、`MGSM` 与
`LiveCodeBench release_latest`。Generation contract 是 seed `20260707`、greedy
pass@1、response cap `3072`、six-way deterministic hash sharding、exact-cover merge、
zero duplicate identities。Generic benchmark 必须 stage-by-stage 跑完六个 shards 后再进入下一项；
LiveCodeBench 使用 canonical 1,055-row six-way sharder。

`PRIMARY` code protocol 使用 raw no-chat HumanEval+、parser-v2、privilege-dropped
sandbox 与 corrected LiveCodeBench。`HERITAGE` chat-protocol HumanEval+ 继续独立保留，
用于解释历史结果，但不得混入 PRIMARY code aggregate。

## Owner Reporting Amendment: Retire Hard Averages

项目负责人在 2026-07-18 明确撤销 `OOD-8` 与 `OOD-category` 两个 human-readable hard
averages。Across-benchmark equal weighting 没有足够的 scientific justification：不同 benchmark
的样本数、difficulty、capability construct 与 score calibration 都不同，一个简单 scalar 会掩盖
真实 capability profile。

因此 current monitor、running summaries 与 final report 必须以 individual benchmark scores 为主，
仅保留语义明确的 within-category summaries（`CodeAvg`、`Reasoning`、`Language`）并同时展示其
components。Legacy machine-readable JSON fields 暂不破坏，以维护 backward compatibility；它们
不得作为当前 scientific conclusion 或 headline metric。

## Preflight And Verification

Launch 前完成了以下 bounded checks：

- TriGLU 与 baseline 的 global steps `158/196/226/256/294` exact checkpoints 全部存在；
- 每个 checkpoint 的 six-rank actor model/optimizer/extra state identity validation 通过；
- TriGLU-196、TriGLU-294、baseline-196、baseline-294 retained exports 可读；
- 两个 imported OOD sources 与 corrected HumanEval+ sources 均完成 source receipt verification；
- launcher、dedicated monitor 与 legacy unified monitor 的 `bash -n` 通过；
- YAML contract 与 focused local/remote tests 通过，最后一轮 remote startup regression suite 为
  `6/6 PASS`；
- launch 前 six GPUs idle；remote data disk 约 `127 GB` free，per-cell `40 GB` guard 生效。

Implementation history 已 commit and push：

- `3db6dcc3` - plan major-checkpoint Other eval wave;
- `82204932` - add paired major-checkpoint Other eval wave;
- `39fed2cb` - fix strict-shell Other eval startup.

Remote 没有 `.git`，所以 source 通过 verified tar archive 同步：

- `qwen_other_eval_8220493.tar`, SHA-256
  `98b3a6bdb3af93515a00ea015517671e29e80389d67c15211dff0eb3f22cbc73`;
- `qwen_other_eval_39fed2c.tar`, SHA-256
  `91e4f1fc05ba136c1f48b4514982b974779b27b9afebd616927ebd1c019e5e75`.

被覆盖的 remote source 在同步前逐个 blob-matched，并归档到
`logs/source_sync_backups/pre_8220493_existing_sources_20260718.tar.gz`。

## Startup Failure And Repair

第一次 screen launch 在任何 import、model export 或 GPU allocation 之前 fail-fast：Bash
`set -u` 遇到 same-line dependent `local` declaration，报错
`label: unbound variable`。该 attempt 的 GPU memory 保持 `0 MiB`，没有生成 predictions，
也没有修改 checkpoint、optimizer、RNG、dataset cursor 或 scientific state。

修复不是只 patch 单一行，而是 audit 并拆分所有同类 same-line dependent declarations，加入
startup regression test 后以 commit `39fed2cb` 同步。Historical failure lines 保留在
`controller.log` 作为 provenance；successful restart 已清除 active `WAVE_FAILED` marker，避免 monitor
把 historical error 误报成 current failure。

## Successful Launch Evidence

Successful autonomous wave 已在 detached screen
`qwen_other_majorsteps6_20260718_v1` 中运行。Launch verification snapshot：

```text
PHASE=EVAL
CELL=triglu_step158
BENCHMARK=gpqa_diamond
STAGE_INDEX=1
STAGE_COUNT=9
controller=running
```

TriGLU-158 temporary export 已 exact-merge，fresh stage 1/9 `GPQA-Diamond` 开始执行；
六张 GPU 当时各约使用 `25.8 GiB`，利用率约 `38-43%`，data disk 约 `124 GB`
free。没有 active `WAVE_FAILED`、OOM、CUDA error 或 model-dispatch error。

两个 imported PRIMARY results 已在 monitor 中显示并与 fresh cells 同表按 global step 排列：

| Cell | HumanEval+ | CodeAvg | Reasoning | Language |
|---|---:|---:|---:|---:|
| baseline-196 | 60.976 | 32.980 | 28.630 | 43.918 |
| TriGLU-294 | 60.976 | 33.439 | 31.285 | 40.483 |

这些 imported numbers 是 completed evidence 的 presentation reuse，不是本轮重新运行的结果。
Fresh cells 未 exact-merge 前只能显示带 numerator/denominator 的 `partial`，不得冒充 final。

第一次 post-launch bounded heartbeat 随后确认 queue 已自动从 stage 1 推进到 stage 2：

```text
CELL=triglu_step158
BENCHMARK=mmlu_pro
STAGE_INDEX=2
STAGE_COUNT=9
GPQA-Diamond=26.766% (final exact merge)
GPU utilization=52-68%
GPU memory=26.7-27.4 GiB per GPU
disk free=124G
```

这证明至少一个 fresh benchmark 已完成 six-way merge，serial controller 没有停在 export 或
stage transition。`MMLU-Pro` 当时仍是 `0/12032 pending`，因此未报告 speculative partial score。

### Step-158 Paired Cell Milestone

后续 bounded verification 确认 first paired boundary 已完整结束，controller 自动推进到
`TriGLU-196 / GPQA-Diamond / stage 1/9`。Step-158 PRIMARY results 为：

| Cell | HumanEval+ | MBPP | LCB | CodeAvg | GPQA | MMLU-Pro | Reasoning | Language |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TriGLU-158 | 59.146 | 30.401 | 10.995 | 33.514 | 26.766 | 33.991 | 30.378 | 44.837 |
| baseline-158 | 59.756 | 29.598 | 9.764 | 33.039 | 28.283 | 34.432 | 31.358 | 43.572 |

At this boundary，TriGLU 的 `CodeAvg` 高 `+0.475` point、Language 高 `+1.265`，
而 baseline 的 Reasoning 高 `+0.980`。这些只是 one-checkpoint paired observations，尚不能
替代完整 trajectory interpretation。Monitor 当时以 completed-cell mean `0.67 h` 估计剩余
六个 fresh cells 约 `4.03 h`；该 ETA 会随每个 cell completion 自动更新。

## Human-Readable Monitoring

项目负责人在 remote shell 运行一次以下命令即可查看 current cell、benchmark/stage、partial/final
scores、elapsed/ETA、GPU、disk 与 recent actionable errors：

```bash
cd /root/autodl-tmp/qwen3_1p7b_single_layer_rl
bash scripts/monitor_qwen3_1p7b_other_eval_majorsteps_6x5090_20260718_v1.sh
```

Legacy unified monitor 也会在 result root 存在时嵌入这张 current wave table。Monitor 默认先显示
PRIMARY complete/partial table，再显示 individual benchmark detail 与 separately labeled HERITAGE
table；不显示 `OOD-8` 或 `OOD-category`。

### 2026-07-18 HumanEval+ Presentation Correction

项目负责人发现 initial dedicated monitor 把 PRIMARY `HumanEval+` 单独摆在 `CodeAvg / Reasoning /
Language` category summaries 旁边，却没有在 individual benchmark detail 中列出。Internal scoring
从未分类错误：`CodeAvg = (HumanEval+ + MBPP + LiveCodeBench) / 3` 一直成立；问题只在
presentation hierarchy。

Current monitor 已改为 category table 只显示 `CodeAvg / Reasoning / Language`，individual table
则明确并列 `HumanEval+ / MBPP / LiveCodeBench` 与其余五项 benchmarks。HERITAGE
chat-protocol HumanEval+ 继续独立显示。该修复不修改任何 score、prediction 或 active evaluation
state。

项目负责人随后确认 `gpqa_diamond` 的 monitor label 不得缩写成可能指向 full GPQA 的 `GPQA`。
Current-stage display 与 individual benchmark header 均明确使用 `GPQA-Diamond`；dataset key、
198-row identity contract 与 historical scores 未发生变化。

## 2026-07-18 GPQA-Diamond Protocol And Parser Audit

项目负责人注意到本 pipeline 的 `GPQA-Diamond` 大多落在约 `20-30%`，明显高于论文
[Is One Layer Enough?](https://arxiv.org/pdf/2607.01232) Table 13 对 Qwen3-1.7B 报告的
Base `5.6%`、Full RL `5.0%`、Layer-10 `5.0%` 与 Layer-12 `7.1%`。本节因此对已经冻结的
predictions 做 read-only audit；没有重新 generation、没有修改 checkpoint，也没有改动 active
evaluator。

### Current Protocol

- Evaluator: `EvalScope 1.8.1`；dataset 为 `AI-ModelScope/gpqa_diamond`，共 `198/198`
  unique rows；
- zero-shot CoT prompt 要求最后一行输出 `ANSWER: [LETTER]`；
- greedy decoding: `do_sample=false`、temperature `0`、seed `20260707`、response cap `3072`；
- four choices 会由 GPQA adapter 内部 deterministic shuffle，随后 target letter 根据同一 mapping
  生成。此前 identity audit 已确认 `198/198` prediction-to-metadata unique mapping，且题目覆盖
  physics、chemistry、biology；未发现 hidden math subset、label leakage 或 scorer/target mismatch。

### Parser Sensitivity

EvalScope `MultiChoiceAdapter.extract_answer` 先尝试 strict/loose `ANSWER:` regex；若均失败，
`_fallback_parse_answer` 会从输出尾部反向寻找 uppercase letter。这个 fallback 不限制为 valid
`A-D`，也不要求它出现在 answer declaration 中。因此它通常只是把 invalid letter 判错，但偶尔也会
把 truncated reasoning 内最后出现的 `A-D` 偶然匹配为 target，形成 parser-sensitive false
positive。Frozen-output audit 的 scorer consistency mismatch 为 `0`：所有 official correct 都确实满足
`extracted_prediction == target`；风险发生在 extraction semantic，而不是 scorer arithmetic。

| Cell | Official correct | Official score | Strict/loose/fallback rows | Fallback correct | `ANSWER:`-only correct | `ANSWER:`-only score | Sensitivity upper bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| untuned base | 41/198 | 20.707% | 131/63/4 | 0 | 41/198 | 20.707% | +0.000 pt |
| TriGLU-158 | 53/198 | 26.768% | 107/73/18 | 2 | 51/198 | 25.758% | +1.010 pt |
| baseline-158 | 56/198 | 28.283% | 105/70/23 | 3 | 53/198 | 26.768% | +1.515 pt |
| baseline-196 | 45/198 | 22.727% | 105/70/23 | 1 | 44/198 | 22.222% | +0.505 pt |
| TriGLU-294 | 59/198 | 29.798% | 120/59/19 | 6 | 53/198 | 26.768% | +3.030 pt |

`ANSWER:`-only 是 intentionally conservative diagnostic，不是新的 official score：少数 fallback-correct
rows 实际包含 valid `\boxed{C}` / `\boxed{B}` answer，不能一概称为 false positive；另一方面，manual
end-snippet review 也确认若干 truncated outputs 确实由 arbitrary-uppercase fallback 偶然判对。所以上表
最后一列是 parser sensitivity 的 upper bound，而不是已经证明的净 inflation。即使删除全部
fallback-correct rows，也无法解释与 paper `5-7%` 的主体差距。

### Statistical Interpretation And Paper Comparability

四选一任务只有在系统始终输出 approximately uniform valid `A-D` 时才有 `25%` random-guess
baseline；若 invalid/unparseable outputs 直接记零，measured accuracy 完全可以低于 `25%`。在
`n=198` 下，random guessing 的 normal-approximation 95% interval 约为
`18.969%-31.031%`。本 pipeline 当前所有 audited scores 都落在该区间内；最高的
TriGLU-294 `59/198 = 29.798%` 是上表 **currently audited cells 中的最高分**，必须作为 strongest
observed result 明确保留。它对 chance 的 exact one-sided binomial `p = 0.07177`：尚未跨过 conventional
`p < 0.05` threshold，但已经构成 **suggestive above-chance statistical signal**，不能被简化成
“没有 signal”或“不可解读”。作为参考，`53/198` 与 `56/198` 的 one-sided p-values 分别约为
`0.3076` 与 `0.1622`。多个 checkpoints 持续落在或高于 `25%` 也可作为 trajectory-level evidence
观察；不过它们共享 model lineage、evaluation set 与训练过程，彼此 correlated，不能伪装成 independent
seed replications 后直接合并 p-value。

这里的 p-value 定义必须写清：在 null hypothesis `H0: 每题以 0.25 概率答对` 下，令
`X ~ Binomial(198, 0.25)`，则 TriGLU-294 的 one-sided p-value 为

```text
P(X >= 59 | H0) = sum[i=59..198] C(198,i) * 0.25^i * 0.75^(198-i)
                = 0.07177
```

它表示 pure random guessing 重复进行这种 198-row experiment 时，偶然得到至少 `59` 个 correct 的
概率约为 `7.177%`；它**不是** `H0` 为真的概率，也不是“模型只有 `7.177%` 概率具备能力”。
`p < 0.05` 只是预先约定的 conventional decision threshold，不是 evidence 的自然断崖：`0.049`
与 `0.051` 不应被叙述成 qualitatively opposite scientific realities。

### Statistical Claim Discipline

- 可以写：TriGLU-294 `29.798%` 是当前 audited results 的最高分，并给出 suggestive above-chance
  evidence；
- 可以写：current GPQA protocol 产生可解释的 multiple-choice measurement，而 paper 的 `5-7%`
  很可能受 undisclosed protocol effects 主导；
- **不可以写：** TriGLU-294 已达到 `p < 0.05` statistical significance，或已经稳健证明
  graduate-science capability；
- **不可以写：** `p = 0.07177` 是 null hypothesis 为真的概率，或将 correlated checkpoints
  当成 independent seeds 合并 significance。

因此，当前最诚实的 interpretation 是：这些数字是 **protocol-specific but interpretable
multiple-choice measurements**。接近或高于 `25%` 说明 current prompt/parser pipeline 正常地把模型的
option selection 映射成 score；尤其 TriGLU-294 对 above-chance capability 给出了有意义但尚未达到
conventional 5% threshold 的 directional evidence。现有 evidence 暂不足以把 `29.798%` 宣称成已经
稳健确立的 graduate-science capability estimate，但也绝不能把它降格为 zero-information chance result。

相比之下，paper 的 `5-7%` 远低于 four-choice random baseline，更可能受到 answer-format failure、
prompting/harness 或 scorer protocol 的强烈影响，因此未必比 current result 更接近 conventional MCQ
accuracy。Paper 没有披露足以复现其 exact prompt、parser、decoding、stop 与 evaluator framework 的
配置；上述原因仍是 inference，不是已证实事实。直到完成 separately named protocol matrix 前，本报告
不得把 current score 与 paper score 当作 apples-to-apples architecture comparison。

### Deferred Diagnostic

若后续项目负责人批准，可在不占 GPU 的情况下对 frozen predictions 增加 separately named
`GPQA-Diamond strict-final-answer` diagnostic：只接受预先批准的 explicit final forms（例如
`ANSWER: X`，以及是否接受 `\boxed{X}` 需单独冻结 policy），绝不 fallback 到 arbitrary uppercase。
Official EvalScope score 必须原样保留；strict diagnostic 只能并列展示，不能 silently overwrite
historical results。

## Free-Response Benchmark Taxonomy And Next-Stage Motivation

项目负责人进一步追问：除了 four-choice `GPQA-Diamond` 与 ten-choice `MMLU-Pro`，是否存在不提供
choices、要求模型直接给出 numerical or textual answer 的 science evaluation。Read-only literature
check 得到以下结论：

| Candidate | Format | Size | Domain | Scoring implication |
|---|---|---:|---|---|
| Official GPQA Extended/Main/Diamond | four-choice | 546/448/198 | graduate biology, physics, chemistry | 25% idealized random-choice baseline |
| `GPQA-Diamond-Freeform-126` | open response, choices removed | 126 | human-curated subset of GPQA-Diamond | reference-aware semantic answer matching; answers are not necessarily numeric |
| `MMLU-Pro-Freeform-493` | open response, choices removed | 493 | broad academic reasoning | reference-aware semantic answer matching |
| `SciBench` | open-ended numerical response | 869 | college mathematics, physics, chemistry | single-number final answers permit deterministic numeric/tolerance scoring |

Official GPQA 没有 ten-choice 或 native free-response parallel variant；Extended、Main 与 Diamond
只是 nested quality/difficulty subsets，并且全部是 four-choice。2025 年的
[Answer Matching Outperforms Multiple Choice](https://arxiv.org/abs/2507.02856) 对 GPQA-Diamond
与 MMLU-Pro 做了 human-curated free-response filtering，留下 `126` 与 `493` 个满足 question
specificity 与 single-unique-answer criteria 的 rows；对应公开数据是
[`nikhilchandak/freeform-datasets`](https://huggingface.co/datasets/nikhilchandak/freeform-datasets)。
该 protocol 能移除 option-recognition shortcut，但因为答案可能是 concept、entity、formula 或
explanation，不能用 arbitrary exact-string parser，必须冻结 reference-aware matcher 与 audit policy。

[SciBench](https://arxiv.org/abs/2307.10635) 更直接对应项目负责人提出的“模型自己回答一个数字”：其
college-level textbook questions 覆盖 mathematics、physics、chemistry，并为 automated evaluation
聚焦 single numerical final answers。它没有 `25%` MCQ floor，能够补充判断模型究竟只是从 options
中 recognize plausible answer，还是能 independently derive quantitative result。这里的“没有 MCQ
floor”不等于数学意义上的 absolute zero chance；numeric tolerance、units、scientific notation、
equivalent forms 与 malformed output 仍必须成为显式 protocol contract。

因此，建议下一阶段保留 current official `GPQA-Diamond` 与 `MMLU-Pro` anchors，同时新增两个
separately named protocols。项目负责人决定 **优先 `GPQA-Diamond-Freeform-126`**：它只有 `126` rows，能以
较低 generation cost 最快检验 current MCQ signal 是否迁移到 generative answer；`SciBench` 的 `869`
rows 放在其后，作为更全面的 quantitative-science extension。它们不得覆盖或回填 current MCQ scores，
也不得与 official paper result 混称同一 benchmark。当前只记录 scientific motivation 与 candidate
protocol；没有下载 dataset、没有修改 evaluator、没有占用 GPU，也没有获得 launch authorization。

## Closeout Gates Still Open

- 8 fresh cells 必须逐项 exact six-shard merge；
- 10/10 cells 必须有 summary、completion/import receipt 与 no-duplicate audit；
- final PRIMARY/HERITAGE tables 与 partial-to-final transition 必须复核；
- compact metrics、logs、manifests 与 final record 必须拉回 local source of truth；
- final closeout update 必须 commit and push，届时本记录才可从 `ACTIVE` 改为 `COMPLETE` 或
  honest degraded/failure status。

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** deliberately deferred。本轮扩展同一 all-model vLLM
  protocol 的 checkpoint coverage，但不构成 HF-vs-vLLM parity matrix closeout。
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deliberately deferred。本轮只 evaluation
  historical checkpoints，不改变 architecture dtype policy。
- **PENDING-03 Registered SHS CausalLM Route:** deliberately deferred。SHS 不在本轮 paired
  TriGLU-baseline grid 内。
