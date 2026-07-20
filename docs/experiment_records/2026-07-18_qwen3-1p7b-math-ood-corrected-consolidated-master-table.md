# Qwen3-1.7B Math + OOD Corrected Consolidated Master Table

Date: 2026-07-18

Status: **CORRECTED CONSOLIDATED SNAPSHOT; MBPP REMAINS PROVISIONAL**

## Purpose / Scope

项目负责人要求把现有 **RL-only** Math 与 OOD results 按原论文 Table 13 的 scan-friendly 形式合并了。为保证每一行同时拥有 Math 与 OOD coverage，本表只列出共同完成全套 evaluation 的 matched checkpoints：steps `158/196/226/256/294`，覆盖 GRPO TriGLU 与 whole-layer baseline，并分别给出 across-checkpoint `mean +/- population std`。早期 steps `20/30/60/98/128` 不进入这张 consolidated table；SFT architectures 与 untuned base 也明确排除。本表不包含论文的 layer-contribution `C`，也不构造 owner 已否定的 hard `Overall` / `OOD-8` average；`MathAvg`、`CodeAvg`、`ReasAvg` 与 `LangAvg` 只作为透明、等权的 category summaries。

所有数字均为 percentage points。`--` 表示该 RL checkpoint 没有执行该 benchmark，而不是零分；GRPO step 是 global update count。

## Consolidated Table

| Setting | Stage | Step | MATH-500 | GSM8K | OlympiadBench | AMC 2023 Avg@32 | MathAvg | AMC 2023 greedy (not avg) | HumanEval+ corrected | MBPP* | LiveCodeBench corrected | CodeAvg* | GPQA-Diamond | MMLU-Pro | ReasAvg | GPQA-Diamond-Freeform manual (not avg) | C-Eval | IFEval | MGSM | LangAvg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TriGLU | GRPO | 158 | 69.200 | 82.942 | 26.815 | 33.359 | 53.079 | 40.000 | 59.146 | 30.401 | 10.995 | 33.514 | 26.766 | 33.991 | 30.378 | 5.556 | 46.582 | 28.513 | 59.417 | 44.837 |
| TriGLU | GRPO | 196 | 66.800 | 82.714 | 28.296 | 33.672 | 52.871 | 37.500 | 61.585 | 28.601 | 10.807 | 33.664 | 26.768 | 34.259 | 30.513 | 4.762 | 47.399 | 26.834 | 57.564 | 43.932 |
| TriGLU | GRPO | 226 | 66.200 | 82.638 | 29.778 | 33.828 | 53.111 | 37.500 | 60.976 | 30.599 | 10.903 | 34.159 | 25.254 | 34.101 | 29.677 | 5.556 | 47.548 | 26.624 | 60.146 | 44.773 |
| TriGLU | GRPO | 256 | 68.800 | 83.548 | 26.370 | 32.109 | 52.707 | 37.500 | 61.585 | 27.600 | 11.376 | 33.521 | 24.242 | 34.353 | 29.297 | 7.937 | 48.218 | 27.673 | 59.566 | 45.152 |
| TriGLU | GRPO | 294 | 66.600 | 82.866 | 29.037 | 34.375 | 53.219 | 37.500 | 60.976 | 29.200 | 10.142 | 33.439 | 29.800 | 32.770 | 31.285 | 7.937 | 47.990 | 27.460 | 46.000 | 40.483 |
| TriGLU mean +/- pop. std | GRPO summary | -- | 67.520 +/- 1.230 | 82.942 +/- 0.322 | 28.059 +/- 1.294 | 33.469 +/- 0.755 | 52.997 +/- 0.184 | 38.000 +/- 1.000 | 60.854 +/- 0.896 | 29.280 +/- 1.121 | 10.845 +/- 0.401 | 33.659 +/- 0.260 | 26.566 +/- 1.880 | 33.895 +/- 0.576 | 30.230 +/- 0.691 | 6.349 +/- 1.328 | 47.548 +/- 0.566 | 27.421 +/- 0.669 | 56.538 +/- 5.340 | 43.836 +/- 1.724 |
| Whole-layer baseline | GRPO | 158 | 66.800 | 82.942 | 26.370 | 33.750 | 52.465 | 35.000 | 59.756 | 29.598 | 9.764 | 33.039 | 28.283 | 34.432 | 31.358 | 6.349 | 45.392 | 26.416 | 58.909 | 43.572 |
| Whole-layer baseline | GRPO | 196 | 67.200 | 82.032 | 24.889 | 32.656 | 51.694 | 37.500 | 60.976 | 28.202 | 9.764 | 32.980 | 22.727 | 34.533 | 28.630 | 4.762 | 45.692 | 27.046 | 59.017 | 43.918 |
| Whole-layer baseline | GRPO | 226 | 67.400 | 82.638 | 25.778 | 35.156 | 52.743 | 32.500 | 60.366 | 29.600 | 10.046 | 33.337 | 25.250 | 34.027 | 29.638 | 3.175 | 46.284 | 25.998 | 58.726 | 43.669 |
| Whole-layer baseline | GRPO | 256 | 66.800 | 81.729 | 25.037 | 33.828 | 51.848 | 40.000 | 57.927 | 28.800 | 9.667 | 32.131 | 27.778 | 34.466 | 31.122 | 3.968 | 46.731 | 27.884 | 58.838 | 44.484 |
| Whole-layer baseline | GRPO | 294 | 66.200 | 82.638 | 25.481 | 35.156 | 52.369 | 32.500 | 58.537 | 27.401 | 9.670 | 31.869 | 28.788 | 34.143 | 31.465 | 4.762 | 46.808 | 26.417 | 59.309 | 44.178 |
| Whole-layer baseline mean +/- pop. std | GRPO summary | -- | 66.880 +/- 0.412 | 82.396 +/- 0.446 | 25.511 +/- 0.533 | 34.109 +/- 0.950 | 52.224 +/- 0.393 | 35.500 +/- 2.915 | 59.512 +/- 1.131 | 28.720 +/- 0.844 | 9.782 +/- 0.139 | 32.671 +/- 0.567 | 26.565 +/- 2.273 | 34.320 +/- 0.198 | 30.443 +/- 1.121 | 4.603 +/- 1.053 | 46.182 +/- 0.560 | 26.752 +/- 0.657 | 58.960 +/- 0.199 | 43.964 +/- 0.334 |

## Aggregate Definitions

- `MathAvg = mean(MATH500, GSM8K, OlympiadBench, AMC Avg@32)`；它明确使用 sampled `AMC Avg@32`，**不使用 AMC greedy**。因此 AMC greedy 放在 MathAvg 之后并以 `not avg` 标记，但仍属于 Math category。
- `CodeAvg* = mean(corrected HumanEval+, provisional MBPP, corrected LiveCodeBench)`；因为 MBPP 尚未完成 full corrected rerun，整个 CodeAvg 也必须带 provisional boundary。
- `ReasAvg = mean(GPQA-Diamond MCQ, MMLU-Pro)`；它**不包含 GPQA-Diamond-Freeform**。因此 `GPQA-Diamond-Freeform manual` 放在 ReasAvg 之后并以 `not avg` 标记，但仍属于 Reasoning category 的 complementary strict-freeform diagnostic。
- `LangAvg = mean(C-Eval, IFEval, MGSM)`；没有再计算一个跨 category 的 Overall。
- Summary rows 的全部 columns 均使用 matched steps `158/196/226/256/294` 这 5 个 checkpoints；standard deviation 是 population std (`ddof=0`)，不是 independent-seed uncertainty estimate。

## Benchmark Guide And Relative Difficulty

下述 difficulty 只是在同一 task family 内的 qualitative comparison；不同 benchmark 的 answer format、chance floor、execution harness 与 language burden 不同，不能把百分比分数直接当成一条 universal difficulty scale。

| Benchmark | What it measures | Relative difficulty / caveat |
|---|---|---|
| MATH-500 | 500 道 competition-style mathematics，覆盖 algebra、geometry、number theory、calculus 等，并要求生成可验证 final answer | Math group 中通常属于 hard；显著难于 GSM8K，和 OlympiadBench 的具体难易会随题型而交叉 |
| GSM8K | English grade-school arithmetic word problems，重点是 multi-step calculation 与文字条件解析 | Math group 中通常最容易；主要测试基础 arithmetic reliability，而非 olympiad insight  |
| OlympiadBench | Olympiad-style mathematical reasoning；当前 pipeline 将其作为高难 Math benchmark | Math group 中通常最难或接近最难，proof-like / long-horizon reasoning burden 高于 AMC 2023  |
| AMC 2023 Avg@32 | 2023 AMC competition questions；每题 32 次 sampled responses 后取平均 accuracy | 难度高于 GSM8K、通常低于 OlympiadBench；`Avg@32` 还额外测量 distribution-wide reliability  |
| AMC 2023 greedy | 同一 AMC 2023 question set 的 deterministic greedy pass@1 | 题目难度不变，但 metric 只看 modal path；它是 Math diagnostic，明确不进入 MathAvg  |
| HumanEval+ | Python function completion，并用 stronger augmented tests 执行验证 | Code group 中等；通常比 MBPP 更严格，但问题规模比 LiveCodeBench 小 |
| MBPP | Mostly Basic Python Problems，要求实现短小 Python functions | 概念上通常是 Code group 最容易；当前 score 受 severe cap/protocol issue 影响，仍为 provisional  |
| LiveCodeBench | Recent contamination-resistant programming problems，要求完整 algorithmic solution 并通过 tests | Code group 通常最难，更接近 competitive programming 与真实 unseen coding evaluation  |
| GPQA-Diamond | Graduate-level science expert questions，当前 protocol 为 four-choice multiple choice | Specialist reasoning 很难，但 four-choice 存在 25% chance floor；不能只按 raw percentage 与 freeform 横比 |
| MMLU-Pro | Broad multi-domain advanced knowledge/reasoning multiple choice，通常有更多 answer options 且比 original MMLU 更难 | Broad hard benchmark；专业深度通常不及 GPQA-Diamond，但覆盖范围更广 |
| GPQA-Diamond-Freeform | 同一 GPQA-Diamond corpus 去掉 choices，要求直接生成答案，并由 manual strict audit 判定 | 比 GPQA-Diamond MCQ 更难，因为没有 option recognition / guessing support；属于 Reasoning diagnostic，但不进入 ReasAvg  |
| C-Eval | Chinese multi-subject exam questions，覆盖 school、university 与 professional knowledge | Mixed difficulty；既测知识与 reasoning，也测 Chinese-language competence，不能简化成纯数学 benchmark  |
| IFEval | Verifiable instruction-following constraints，例如 format、length、keywords 与 structural compliance | 不主要测知识难度，而测 constraint compliance；和其她 Language columns 不共享单一难度轴 |
| MGSM | GSM8K-style grade-school math translated across multiple languages | 数学本体接近 GSM8K，但 multilingual transfer 增加额外 burden；通常比 English-only GSM8K 更难 |

## Why Corrected Values Were Needed

1. **HumanEval+**：legacy chat-wrapped prompt 触发 Hebrew/template collapse 与 severe interface mismatch；corrected route 使用 raw EvalScope instruction、no chat wrapper、`humanevalplus_parser_v2_fixed` 与 fixed sandbox。它修复 evaluator protocol，但不 claim exact paper parity。
2. **LiveCodeBench**：历史全零来自 sandbox `output` contract mismatch，不是模型突然失去全部 coding ability；表中使用 frozen generations 在 fixed contract 下的 corrected review score。
3. **AMC labels**：旧 monitor 曾把 greedy pass@1 折叠进 sampled Avg@32；表中明确拆成两列，stored generations/reviews 不变。baseline step20 的 AMC@32 score 是 `295/1279`，因为一条 generated row 缺少 score-bearing review；没有伪造为 1280 denominator。
4. **GPQA-Diamond-Freeform**：表中采用完整 `126 questions x 10 cells = 1260 responses` manual row-by-row audit；`uncertain` 不计 correct。它与 four-choice GPQA-Diamond 是不同 protocol。
5. **MBPP asterisk**：目前只有 heritage/current score 与 bounded replay evidence，尚无覆盖全部 cells 的 protocol-matrix/full rerun。为了完整保留 existing result，本表显示它，但绝不称其为 fully corrected；任何依赖它的 `CodeAvg*` 同样 provisional。

## Cap-Hit Ledger

| Benchmark / protocol | Cap hits | Rows | Rate | Interpretation |
|---|---:|---:|---:|---|
| MATH-500 (RL checkpoint grid) | 251 | 5,000 | 5.02% | paper-style 3072 cap; retained |
| GSM8K (RL checkpoint grid) | 57 | 13,190 | 0.43% | paper-style 3072 cap; retained |
| OlympiadBench (RL checkpoint grid) | 935 | 6,750 | 13.85% | paper-style 3072 cap; retained |
| AMC Avg@32 (RL checkpoint grid) | 491 | 12,800 | 3.84% | paper-style 3072 cap; retained |
| AMC greedy pass@1 (RL checkpoint grid) | 33 | 400 | 8.25% | paper-style 3072 cap; retained |
| HumanEval+ corrected raw no-chat | 41 | 1,640 | 2.50% | targeted continuation pending |
| MBPP heritage | 4,696 | 5,000 | 93.92% | protocol matrix + full rerun pending |
| LiveCodeBench corrected | 1,027 | 10,550 | 9.73% | targeted continuation pending |
| GPQA-Diamond MCQ | 195 | 1,980 | 9.85% | targeted continuation pending |
| MMLU-Pro | 5,686 | 120,320 | 4.73% | targeted continuation pending |
| C-Eval | 9,195 | 13,460 | 68.31% | protocol matrix + full rerun pending |
| IFEval | 2,115 | 5,410 | 39.09% | targeted continuation pending |
| MGSM | 2,369 | 27,500 | 8.61% | targeted continuation pending |
| GPQA-Diamond-Freeform | 96 | 1,260 | 7.62% | targeted continuation pending |

所有 Math benchmarks 在现有 RL checkpoint grid 中都出现过 cap hit，GRPO training rollouts 的 logs 里也存在 nonzero `response_length/clip_ratio`。目前 Math 主表姑且保留 paper-style cap-hit-inclusive scores，不做 post-hoc continuation 后静默覆写。

对 Math 而言 cap-hit fraction 整体不占多数，其中人工抽查的许多 case 是无意义 repetition loop；不过不能排除少量 response 若继续生成，最终会到达正确答案。因此当前结果适合比较同 protocol 下的 trajectory，不应被描述成 uncapped capability ceiling。

这个“占比不多”不能错误外推到全部 OOD：MBPP `93.92%`、C-Eval `68.31%`、IFEval `39.09%` 是明显 exceptions。MBPP/C-Eval 必须先做 protocol matrix 再 full rerun；其余 OOD benches 才适合 targeted prefill continuation。GPQA-Freeform 原始 96 个 cap-hit rows 已全部 manual-audited 为 incorrect，但未来 replacement rows 仍需 targeted manual re-audit。

## Interpretation Boundary

- 表内 corrected 是“使用目前已经完成且有 provenance 的 correction”；它不等于所有 benchmark 已达到 paper-exact parity。
- Across-checkpoint std 描述同一 training trajectory 上 checkpoints 的波动，不是 multiple independent seeds；不得拿它直接 claim statistical significance。
- GPQA-Diamond MCQ 仍是 protocol-specific four-choice score；freeform manual column提供 complementary evidence，但两者不可互换。

## Machine-Readable Artifacts

- `docs/experiment_records/figures/2026-07-18_qwen3_1p7b_math_ood_corrected_master_table.png`：publication-style RL-only corrected master-table PNG。
- `docs/experiment_records/figures/2026-07-18_qwen3_1p7b_math_ood_corrected_master_table.manifest.json`：PNG/source SHA-256、dimensions 与 row-count manifest。
- `docs/experiment_records/compact_metrics/2026-07-18_qwen3_1p7b_math_ood_corrected_master_table.csv`：flat table with numeric metric/std columns。
- `docs/experiment_records/compact_metrics/2026-07-18_qwen3_1p7b_math_ood_corrected_master_table.json`：rows、correction boundaries 与 source provenance。
- `docs/experiment_records/compact_metrics/2026-07-18_qwen3_1p7b_math_rl_checkpoint_snapshot.json`：从远端 authoritative Math review/receipt roots 生成并 SHA-256 verified 的 source snapshot；SHA-256 `c2a3f99762cd3a3e68b4f1de6665e5dff8a1a304b3039c09fc17fdf331ded057`。

## Pending Obligations Carried Forward

Canonical registry: `docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- `PENDING-01 Eval Parity Matrix`：仍为 **PENDING**；本表整理已有 corrections，不构成完整 HF-vs-vLLM matrix closeout。
- `PENDING-02 Pure-BF16 SHS And TriGLU`：仍为 **PENDING**，本表 deferred。
- `PENDING-03 Registered SHS CausalLM Route`：仍为 **PENDING**，本表 deferred。
