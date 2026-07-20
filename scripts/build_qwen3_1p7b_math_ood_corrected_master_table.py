from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "docs" / "experiment_records"
COMPACT = RECORDS / "compact_metrics"
MATH_SNAPSHOT = COMPACT / "2026-07-18_qwen3_1p7b_math_rl_checkpoint_snapshot.json"
OOD_ROOT = COMPACT / "2026-07-18_qwen3_1p7b_existing_ood_cap_evidence"
GPQA_FREEFORM = COMPACT / "2026-07-18_qwen3_1p7b_gpqa_diamond_freeform126_manual_agent_audit.json"
OOD_CAPS = COMPACT / "2026-07-18_ood_cap_hit_scope_summary.json"
OUT_CSV = COMPACT / "2026-07-18_qwen3_1p7b_math_ood_corrected_master_table.csv"
OUT_JSON = COMPACT / "2026-07-18_qwen3_1p7b_math_ood_corrected_master_table.json"
OUT_MD = RECORDS / "2026-07-18_qwen3-1p7b-math-ood-corrected-consolidated-master-table.md"
OUT_PNG = RECORDS / "figures" / "2026-07-18_qwen3_1p7b_math_ood_corrected_master_table.png"
OUT_PNG_MANIFEST = OUT_PNG.with_suffix(".manifest.json")

STEPS = (158, 196, 226, 256, 294)
OOD_STEPS = STEPS
METRICS = (
    "math500",
    "gsm8k",
    "olympiad",
    "amc_avg32",
    "math_avg",
    "amc_greedy",
    "humaneval_plus",
    "mbpp",
    "livecodebench",
    "code_avg",
    "gpqa_diamond",
    "mmlu_pro",
    "reasoning_avg",
    "gpqa_freeform",
    "ceval",
    "ifeval",
    "mgsm",
    "language_avg",
)
HEADERS = {
    "math500": "MATH-500",
    "gsm8k": "GSM8K",
    "olympiad": "OlympiadBench",
    "amc_avg32": "AMC 2023 Avg@32",
    "math_avg": "MathAvg",
    "amc_greedy": "AMC 2023 greedy (not avg)",
    "humaneval_plus": "HumanEval+ corrected",
    "mbpp": "MBPP*",
    "livecodebench": "LiveCodeBench corrected",
    "code_avg": "CodeAvg*",
    "gpqa_diamond": "GPQA-Diamond",
    "mmlu_pro": "MMLU-Pro",
    "reasoning_avg": "ReasAvg",
    "gpqa_freeform": "GPQA-Diamond-Freeform manual (not avg)",
    "ceval": "C-Eval",
    "ifeval": "IFEval",
    "mgsm": "MGSM",
    "language_avg": "LangAvg",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if len(present) == len(values) else None


def empty_row(setting: str, stage: str, step: int | None, row_type: str = "checkpoint") -> dict:
    return {
        "setting": setting,
        "stage": stage,
        "step": step,
        "row_type": row_type,
        "metrics": {metric: None for metric in METRICS},
        "population_std": {metric: None for metric in METRICS},
    }


def fill_derived(row: dict) -> None:
    metrics = row["metrics"]
    metrics["code_avg"] = mean(
        metrics["humaneval_plus"], metrics["mbpp"], metrics["livecodebench"]
    )
    metrics["reasoning_avg"] = mean(metrics["gpqa_diamond"], metrics["mmlu_pro"])
    metrics["language_avg"] = mean(metrics["ceval"], metrics["ifeval"], metrics["mgsm"])


def math_metrics(cell: dict) -> dict[str, float]:
    benchmarks = cell["benchmarks"]
    return {
        "math500": 100 * benchmarks["paper_math500"]["accuracy"],
        "gsm8k": 100 * benchmarks["paper_gsm8k"]["accuracy"],
        "olympiad": 100 * benchmarks["paper_olympiadbench"]["accuracy"],
        "amc_avg32": 100 * benchmarks["paper_amc23"]["accuracy"],
        "amc_greedy": 100 * benchmarks["paper_amc23_greedy"]["accuracy"],
        "math_avg": cell["math_avg"],
    }


def ood_metrics(variant: str, step: int, freeform: dict) -> dict[str, float]:
    cell = read_json(OOD_ROOT / "ood_cells" / f"{variant}_step_{step}.json")
    benchmarks = cell["benchmarks"]
    parser = cell["parser_sensitive_code_comparison"]
    gpqa = freeform["cells"][f"{variant}_step{step}"]
    return {
        "humaneval_plus": 100 * parser["humaneval_plus_prompt_corrected"],
        "mbpp": 100 * benchmarks["mbpp"]["score"],
        "livecodebench": 100 * benchmarks["live_code_bench"]["score"],
        "gpqa_diamond": 100 * benchmarks["gpqa_diamond"]["score"],
        "gpqa_freeform": 100 * gpqa["strict_accuracy"],
        "mmlu_pro": 100 * benchmarks["mmlu_pro"]["score"],
        "ceval": 100 * benchmarks["ceval"]["score"],
        "ifeval": 100 * benchmarks["ifeval"]["score"],
        "mgsm": 100 * benchmarks["mgsm"]["score"],
    }


def rl_rows(math_snapshot: dict, freeform: dict) -> list[dict]:
    rows = []
    for variant, label in (("triglu", "TriGLU"), ("baseline", "Whole-layer baseline")):
        for step in STEPS:
            row = empty_row(label, "GRPO", step)
            row["metrics"].update(math_metrics(math_snapshot["cells"][f"{variant}_step{step}"]))
            if step in OOD_STEPS:
                row["metrics"].update(ood_metrics(variant, step, freeform))
            fill_derived(row)
            rows.append(row)
    return rows


def summary_row(rows: list[dict], setting: str) -> dict:
    result = empty_row(f"{setting} mean +/- pop. std", "GRPO summary", None, "summary")
    selected = [row for row in rows if row["setting"] == setting and row["stage"] == "GRPO"]
    for metric in METRICS:
        values = [row["metrics"][metric] for row in selected if row["metrics"][metric] is not None]
        if values:
            result["metrics"][metric] = statistics.fmean(values)
            result["population_std"][metric] = statistics.pstdev(values)
    return result


def fmt(value: float | None, std: float | None = None) -> str:
    if value is None:
        return "--"
    if std is None:
        return f"{value:.3f}"
    return f"{value:.3f} +/- {std:.3f}"


def markdown_table(rows: list[dict]) -> str:
    columns = ["Setting", "Stage", "Step", *[HEADERS[metric] for metric in METRICS]]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        values = [row["setting"], row["stage"], "--" if row["step"] is None else str(row["step"])]
        values.extend(
            fmt(row["metrics"][metric], row["population_std"][metric])
            if row["row_type"] == "summary"
            else fmt(row["metrics"][metric])
            for metric in METRICS
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def cap_table(math_snapshot: dict, ood_caps: dict) -> str:
    labels = {
        "paper_math500": "MATH-500",
        "paper_gsm8k": "GSM8K",
        "paper_olympiadbench": "OlympiadBench",
        "paper_amc23": "AMC Avg@32",
        "paper_amc23_greedy": "AMC greedy pass@1",
    }
    totals = {key: {"hits": 0, "rows": 0} for key in labels}
    for cell in math_snapshot["cells"].values():
        if cell["step"] not in STEPS:
            continue
        for key in labels:
            evidence = cell["cap_hits"][key]
            totals[key]["hits"] += evidence["cap_hits"]
            totals[key]["rows"] += evidence["rows"]
    lines = [
        "| Benchmark / protocol | Cap hits | Rows | Rate | Interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    for key, label in labels.items():
        evidence = totals[key]
        lines.append(
            f"| {label} (RL checkpoint grid) | {evidence['hits']:,} | {evidence['rows']:,} | "
            f"{100 * evidence['hits'] / evidence['rows']:.2f}% | paper-style 3072 cap; retained |"
        )
    names = [
        ("HumanEval+ corrected raw no-chat", 1640),
        ("MBPP heritage", 5000),
        ("LiveCodeBench corrected", 10550),
        ("GPQA-Diamond MCQ", 1980),
        ("MMLU-Pro", 120320),
        ("C-Eval", 13460),
        ("IFEval", 5410),
        ("MGSM", 27500),
        ("GPQA-Diamond-Freeform", 1260),
    ]
    for name, denominator in names:
        hits = ood_caps["cap_hit_rows"][name]
        action = "protocol matrix + full rerun pending" if name in {"MBPP heritage", "C-Eval"} else "targeted continuation pending"
        lines.append(f"| {name} | {hits:,} | {denominator:,} | {100 * hits / denominator:.2f}% | {action} |")
    return "\n".join(lines)


def write_csv(rows: list[dict]) -> None:
    fields = ["setting", "stage", "step", "row_type"]
    for metric in METRICS:
        fields.extend((metric, f"{metric}_population_std"))
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in ("setting", "stage", "step", "row_type")}
            for metric in METRICS:
                flat[metric] = row["metrics"][metric]
                flat[f"{metric}_population_std"] = row["population_std"][metric]
            writer.writerow(flat)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(rows: list[dict], math_snapshot: dict) -> None:
    assert len(rows) == 12, f"expected 10 matched checkpoints + 2 summaries, got {len(rows)}"
    assert all(row["stage"] != "SFT" for row in rows), "RL-only table contains SFT"
    for setting in ("TriGLU", "Whole-layer baseline"):
        checkpoints = [
            row for row in rows if row["setting"] == setting and row["row_type"] == "checkpoint"
        ]
        assert [row["step"] for row in checkpoints] == list(STEPS)
    for cell in math_snapshot["cells"].values():
        assert all(evidence["cap_hits"] > 0 for evidence in cell["cap_hits"].values())


def main() -> None:
    math_snapshot = read_json(MATH_SNAPSHOT)
    freeform = read_json(GPQA_FREEFORM)
    ood_caps = read_json(OOD_CAPS)

    rl = rl_rows(math_snapshot, freeform)
    rows = []
    for setting in ("TriGLU", "Whole-layer baseline"):
        rows.extend(row for row in rl if row["setting"] == setting)
        rows.append(summary_row(rl, setting))
    validate(rows, math_snapshot)

    payload = {
        "schema_version": 1,
        "date": "2026-07-18",
        "status": "corrected_consolidated_snapshot",
        "percentage_units": True,
        "population_std_ddof": 0,
        "rows": rows,
        "correction_boundaries": {
            "humaneval_plus": "corrected raw EvalScope instruction, no chat, parser-v2, fixed sandbox",
            "livecodebench": "fixed sandbox output contract; frozen generations re-reviewed",
            "mbpp": "current heritage/provisional value; full protocol-matrix correction remains pending",
            "gpqa_freeform": "strict manual row-by-row audit; uncertain rows count as not correct",
            "math": "canonical paper-style 3072-cap scores retained; AMC greedy label separated from Avg@32",
        },
        "sources": {
            "math": str(MATH_SNAPSHOT.relative_to(ROOT)),
            "ood_cells": str(OOD_ROOT.relative_to(ROOT) / "ood_cells"),
            "gpqa_freeform": str(GPQA_FREEFORM.relative_to(ROOT)),
            "cap_summary": str(OOD_CAPS.relative_to(ROOT)),
        },
        "source_sha256": {
            "math": sha256(MATH_SNAPSHOT),
            "gpqa_freeform": sha256(GPQA_FREEFORM),
            "cap_summary": sha256(OOD_CAPS),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(rows)

    report = f"""# Qwen3-1.7B Math + OOD Corrected Consolidated Master Table

Date: 2026-07-18

Status: **CORRECTED CONSOLIDATED SNAPSHOT; MBPP REMAINS PROVISIONAL**

## Purpose / Scope

项目负责人要求把现有 **RL-only** Math 与 OOD results 按原论文 Table 13 的 scan-friendly 形式合并了。为保证每一行同时拥有 Math 与 OOD coverage，本表只列出共同完成全套 evaluation 的 matched checkpoints：steps `158/196/226/256/294`，覆盖 GRPO TriGLU 与 whole-layer baseline，并分别给出 across-checkpoint `mean +/- population std`。早期 steps `20/30/60/98/128` 不进入这张 consolidated table；SFT architectures 与 untuned base 也明确排除。本表不包含论文的 layer-contribution `C`，也不构造 owner 已否定的 hard `Overall` / `OOD-8` average；`MathAvg`、`CodeAvg`、`ReasAvg` 与 `LangAvg` 只作为透明、等权的 category summaries。

所有数字均为 percentage points。`--` 表示该 RL checkpoint 没有执行该 benchmark，而不是零分；GRPO step 是 global update count。

## Consolidated Table

{markdown_table(rows)}

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

{cap_table(math_snapshot, ood_caps)}

所有 Math benchmarks 在现有 RL checkpoint grid 中都出现过 cap hit，GRPO training rollouts 的 logs 里也存在 nonzero `response_length/clip_ratio`。目前 Math 主表姑且保留 paper-style cap-hit-inclusive scores，不做 post-hoc continuation 后静默覆写。

对 Math 而言 cap-hit fraction 整体不占多数，其中人工抽查的许多 case 是无意义 repetition loop；不过不能排除少量 response 若继续生成，最终会到达正确答案。因此当前结果适合比较同 protocol 下的 trajectory，不应被描述成 uncapped capability ceiling。

这个“占比不多”不能错误外推到全部 OOD：MBPP `93.92%`、C-Eval `68.31%`、IFEval `39.09%` 是明显 exceptions。MBPP/C-Eval 必须先做 protocol matrix 再 full rerun；其余 OOD benches 才适合 targeted prefill continuation。GPQA-Freeform 原始 96 个 cap-hit rows 已全部 manual-audited 为 incorrect，但未来 replacement rows 仍需 targeted manual re-audit。

## Interpretation Boundary

- 表内 corrected 是“使用目前已经完成且有 provenance 的 correction”；它不等于所有 benchmark 已达到 paper-exact parity。
- Across-checkpoint std 描述同一 training trajectory 上 checkpoints 的波动，不是 multiple independent seeds；不得拿它直接 claim statistical significance。
- GPQA-Diamond MCQ 仍是 protocol-specific four-choice score；freeform manual column提供 complementary evidence，但两者不可互换。

## Machine-Readable Artifacts

- `{OUT_PNG.relative_to(ROOT).as_posix()}`：publication-style RL-only corrected master-table PNG。
- `{OUT_PNG_MANIFEST.relative_to(ROOT).as_posix()}`：PNG/source SHA-256、dimensions 与 row-count manifest。
- `{OUT_CSV.relative_to(ROOT).as_posix()}`：flat table with numeric metric/std columns。
- `{OUT_JSON.relative_to(ROOT).as_posix()}`：rows、correction boundaries 与 source provenance。
- `{MATH_SNAPSHOT.relative_to(ROOT).as_posix()}`：从远端 authoritative Math review/receipt roots 生成并 SHA-256 verified 的 source snapshot；SHA-256 `{sha256(MATH_SNAPSHOT)}`。

## Pending Obligations Carried Forward

Canonical registry: `docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- `PENDING-01 Eval Parity Matrix`：仍为 **PENDING**；本表整理已有 corrections，不构成完整 HF-vs-vLLM matrix closeout。
- `PENDING-02 Pure-BF16 SHS And TriGLU`：仍为 **PENDING**，本表 deferred。
- `PENDING-03 Registered SHS CausalLM Route`：仍为 **PENDING**，本表 deferred。
"""
    OUT_MD.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
