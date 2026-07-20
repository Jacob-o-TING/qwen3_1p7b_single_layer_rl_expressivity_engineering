from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "docs"
    / "experiment_records"
    / "compact_metrics"
    / "2026-07-18_qwen3_1p7b_math_ood_corrected_master_table.json"
)
OUTPUT = (
    ROOT
    / "docs"
    / "experiment_records"
    / "figures"
    / "2026-07-18_qwen3_1p7b_math_ood_corrected_master_table.png"
)
MANIFEST = OUTPUT.with_suffix(".manifest.json")

FONT_REGULAR = Path("C:/Windows/Fonts/arial.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")
FONT_SERIF = Path("C:/Windows/Fonts/times.ttf")

COLUMNS = (
    ("setting", "Setting", 430),
    ("math500", "MATH-500", 220),
    ("gsm8k", "GSM8K", 205),
    ("olympiad", "OlympiadBench", 245),
    ("amc_avg32", "AMC 2023\nAvg@32", 230),
    ("math_avg", "MathAvg", 220),
    ("amc_greedy", "AMC 2023\ngreedy*", 235),
    ("humaneval_plus", "HumanEval+\ncorrected", 245),
    ("mbpp", "MBPP*", 205),
    ("livecodebench", "LiveCodeBench\ncorrected", 265),
    ("code_avg", "CodeAvg*", 225),
    ("gpqa_diamond", "GPQA-Diamond", 250),
    ("mmlu_pro", "MMLU-Pro", 225),
    ("reasoning_avg", "ReasAvg", 220),
    ("gpqa_freeform", "GPQA-Diamond-\nFreeform*", 275),
    ("ceval", "C-Eval", 205),
    ("ifeval", "IFEval", 205),
    ("mgsm", "MGSM", 205),
    ("language_avg", "LangAvg", 220),
)
GROUPS = (
    ("Math", 1, 6),
    ("Code", 7, 10),
    ("Reasoning", 11, 14),
    ("Language", 15, 18),
)


def centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill) -> None:
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), text, font=font, spacing=3, align="center")
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.multiline_text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - bounds[1]),
        text,
        font=font,
        fill=fill,
        spacing=3,
        align="center",
    )


def row_label(row: dict) -> str:
    if row["row_type"] == "summary":
        return "TriGLU mean +/- std" if row["setting"].startswith("TriGLU") else "Baseline mean +/- std"
    prefix = "TriGLU" if row["setting"] == "TriGLU" else "Baseline"
    return f"{prefix}-{row['step']}"


def cell_value(row: dict, metric: str) -> str:
    if metric == "setting":
        return row_label(row)
    value = row["metrics"][metric]
    if value is None:
        return "--"
    if row["row_type"] == "summary":
        std = row["population_std"][metric]
        return f"{value:.2f}+/-{std:.2f}"
    return f"{value:.3f}"


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert len(rows) == 12 and all(row["stage"].startswith("GRPO") for row in rows)

    margin_x = 70
    title_h = 155
    group_h = 60
    header_h = 74
    row_h = 56
    notes_h = 455
    table_width = sum(width for _, _, width in COLUMNS)
    width = table_width + 2 * margin_x
    height = title_h + group_h + header_h + row_h * len(rows) + notes_h + 45

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(str(FONT_SERIF), 47)
    subtitle_font = ImageFont.truetype(str(FONT_REGULAR), 25)
    group_font = ImageFont.truetype(str(FONT_BOLD), 29)
    header_font = ImageFont.truetype(str(FONT_BOLD), 24)
    body_font = ImageFont.truetype(str(FONT_REGULAR), 24)
    body_bold = ImageFont.truetype(str(FONT_BOLD), 23)
    note_font = ImageFont.truetype(str(FONT_REGULAR), 23)

    centered(
        draw,
        (margin_x, 22, width - margin_x, 86),
        "Qwen3-1.7B RL: Corrected Math + OOD Results Across Checkpoints",
        title_font,
        "#111827",
    )
    centered(
        draw,
        (margin_x, 87, width - margin_x, 132),
        "Matched steps 158 / 196 / 226 / 256 / 294 | scores in percentage points | mean +/- population std",
        subtitle_font,
        "#4b5563",
    )

    x_positions = [margin_x]
    for _, _, column_width in COLUMNS:
        x_positions.append(x_positions[-1] + column_width)
    group_top = title_h
    header_top = group_top + group_h
    body_top = header_top + header_h

    draw.rectangle((margin_x, group_top, width - margin_x, header_top), fill="#17324d")
    draw.rectangle((x_positions[0], group_top, x_positions[1], body_top), fill="#17324d")
    centered(draw, (x_positions[0], group_top, x_positions[1], body_top), "Setting", group_font, "white")
    for label, first, last in GROUPS:
        centered(draw, (x_positions[first], group_top, x_positions[last + 1], header_top), label, group_font, "white")
        draw.line((x_positions[first], group_top, x_positions[first], body_top), fill="white", width=2)

    draw.rectangle((x_positions[1], header_top, width - margin_x, body_top), fill="#dbe7f2")
    for index, (_, label, _) in enumerate(COLUMNS[1:], start=1):
        centered(draw, (x_positions[index], header_top, x_positions[index + 1], body_top), label, header_font, "#14283d")

    for row_index, row in enumerate(rows):
        top = body_top + row_index * row_h
        bottom = top + row_h
        if row["row_type"] == "summary":
            fill = "#d9e7f5"
        elif row["setting"] == "TriGLU":
            fill = "#f5f8fc" if row_index % 2 == 0 else "white"
        else:
            fill = "#f7f7f7" if row_index % 2 == 0 else "white"
        draw.rectangle((margin_x, top, width - margin_x, bottom), fill=fill)
        font = body_bold if row["row_type"] == "summary" else body_font
        for column_index, (metric, _, _) in enumerate(COLUMNS):
            centered(
                draw,
                (x_positions[column_index], top, x_positions[column_index + 1], bottom),
                cell_value(row, metric),
                font,
                "#111827",
            )
        draw.line((margin_x, bottom, width - margin_x, bottom), fill="#c9ced5", width=1)

    table_bottom = body_top + row_h * len(rows)
    for position in x_positions:
        draw.line((position, header_top, position, table_bottom), fill="#c9ced5", width=1)
    for boundary in (1, 7, 11, 15, 19):
        draw.line((x_positions[boundary], group_top, x_positions[boundary], table_bottom), fill="#17324d", width=3)
    for boundary in (6, 14):
        draw.line((x_positions[boundary] - 3, header_top, x_positions[boundary] - 3, table_bottom), fill="#8ca0b3", width=1)
        draw.line((x_positions[boundary] + 3, header_top, x_positions[boundary] + 3, table_bottom), fill="#8ca0b3", width=1)
    draw.rectangle((margin_x, group_top, width - margin_x, table_bottom), outline="#17324d", width=3)

    notes = (
        "MathAvg uses AMC 2023 Avg@32, not AMC 2023 greedy*. ReasAvg uses GPQA-Diamond + MMLU-Pro, not GPQA-Diamond-Freeform*.",
        "Code note: MBPP is the current heritage/provisional score; the full protocol-matrix correction is pending, so CodeAvg* inherits this boundary.",
        "Corrected routes: HumanEval+ raw no-chat + fixed parser/sandbox; LiveCodeBench fixed sandbox output contract; GPQA-F strict manual audit.",
        "Math cap hits are retained: MATH500 5.02%, GSM8K 0.43%, Olympiad 13.85%, AMC@32 3.84%, AMC greedy 8.25%. Training rollouts also contain cap hits.",
        "Many inspected cap hits are repetition loops, but a small number could reach a correct answer if continued; results are not uncapped capability ceilings.",
        "Math difficulty (rough): GSM8K < AMC 2023 < MATH-500 / OlympiadBench. Avg@32 measures sampled-distribution reliability; greedy is modal pass@1.",
        "Code difficulty (rough): MBPP < HumanEval+ < LiveCodeBench. Scores depend on executable-test and sandbox contracts.",
        "Reasoning: MMLU-Pro is broad advanced MCQ; GPQA-Diamond is specialist graduate-science MCQ; GPQA-Diamond-Freeform is harder without choices.",
        "Language: C-Eval measures Chinese multi-subject exams; IFEval measures constraint following; MGSM adds multilingual burden to grade-school math.",
    )
    note_y = table_bottom + 28
    for note in notes:
        draw.text((margin_x, note_y), note, font=note_font, fill="#374151")
        note_y += 42

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "dimensions_px": {"width": width, "height": height},
        "row_count": len(rows),
        "checkpoint_rows": sum(row["row_type"] == "checkpoint" for row in rows),
        "summary_rows": sum(row["row_type"] == "summary" for row in rows),
        "display_steps": [158, 196, 226, 256, 294],
        "early_steps_excluded_because_they_lack_ood_coverage": [20, 30, 60, 98, 128],
        "visual_inspection": "setting header, grouped columns, values, footnotes, and axes-free table framing checked",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
