# 实验记录 / Experiment Records

这个目录用于保存 Qwen3-1.7B 项目的 execution records、benchmark
observations、abnormal events 与 shutdown decisions。它们既是 human-readable
scientific narrative，也是后续 Agent 复盘时的 durable source of truth。

## 写作语言 / Language Contract

所有 plan、record、audit report、amendment 与 closeout 都必须使用自然的
Chinese-English mixed style：用中文讲清楚 motivation、reasoning、interpretation、
owner decision 与 next action，同时保留 architecture、benchmark、metric、config
key、command、path、hash、error string 和 gate label 等 English technical handles。

不允许只把标题改成 bilingual 就算完成，也不允许机械逐词翻译。每个 major
table 附近都要有中英混搭的 interpretation。JSON/JSONL/CSV/manifest 等
machine-readable artifacts 不受此语言规则影响，schema 必须原样保留。

历史 records 采用 dual-edition conservative migration：English original 保留在
原路径且不覆盖；完整的 mixed companion 放在 `bilingual_companions/` 下，使用同名
basename、反向链接、中文导读与原始 evidence。任何新增 amendment 则必须完整遵循
上述 mixed-language contract。

Companion 总索引见
[`bilingual_companions/README.md`](bilingual_companions/README.md)，其中记录每份
English original、对应 mixed companion 与 original SHA-256，便于 verification。

未来若 report 从一份 English draft 开始，先 archive English original，再生成
mixed edition；绝不允许用翻译后的版本 silently overwrite 原件。

文件名继续遵循：

```text
YYYY-MM-DD_<short-experiment-summary>-record.md
```

Experiment plans 仍放在 `docs/experiment_plans/`。Cross-project research notes
只有在直接描述某次 Qwen3 run，并且 provenance 与 encoding 都已确认时，才迁入
这里。
