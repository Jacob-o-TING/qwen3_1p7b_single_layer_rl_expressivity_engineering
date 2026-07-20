from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


EVALSCOPE_INSTRUCTION = (
    "Read the following function signature and docstring, and fully implement "
    "the function described. Your response should only contain the code for "
    "this function.\n"
)

MATRIX_CELLS = (
    "evalscope_chat_instruction_control",
    "evalscope_raw_instruction_nochat",
    "canonical_completion_nochat",
)

_HEBREW = re.compile(r"[\u0590-\u05ff]")


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_source_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"source row {line_number} has no metadata")
            task = {
                "source_index": int(row["index"]),
                "task_id": str(metadata["task_id"]),
                "entry_point": str(metadata["entry_point"]),
                "prompt": str(metadata["prompt"]),
                "test": str(metadata["test"]),
            }
            task_id = task["task_id"]
            if task_id in tasks and tasks[task_id] != task:
                raise ValueError(f"conflicting duplicate task_id: {task_id}")
            tasks[task_id] = task
    if len(tasks) != 164:
        raise ValueError(f"expected 164 unique HumanEval+ tasks, found {len(tasks)}")
    return list(tasks.values())


def select_tasks(tasks: Iterable[dict[str, Any]], *, seed: int, count: int) -> list[dict[str, Any]]:
    rows = list(tasks)
    if not 0 < count <= len(rows):
        raise ValueError(f"sample count must be in [1, {len(rows)}], got {count}")

    def selection_key(task: dict[str, Any]) -> tuple[str, str]:
        task_id = str(task["task_id"])
        digest = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).hexdigest()
        return digest, task_id

    selected = sorted(rows, key=selection_key)[:count]
    return [
        {
            "ledger_index": index,
            "selection_sha256": selection_key(task)[0],
            **task,
        }
        for index, task in enumerate(selected)
    ]


def render_prompt(tokenizer: Any, cell: str, task: dict[str, Any]) -> str:
    if cell not in MATRIX_CELLS:
        raise ValueError(f"unknown matrix cell: {cell}")
    canonical = str(task["prompt"])
    if cell == "canonical_completion_nochat":
        return canonical
    instruction = EVALSCOPE_INSTRUCTION + canonical
    if cell == "evalscope_raw_instruction_nochat":
        return instruction
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def request_seed(base_seed: int, task_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{task_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def classify_generation(
    *, raw: str, parsed: str, canonical_prompt: str, finish_reason: str | None
) -> dict[str, Any]:
    nonempty_lines = [line.strip() for line in raw.splitlines() if line.strip()]
    line_counts = Counter(nonempty_lines)
    dominant_line_count = max(line_counts.values(), default=0)
    dominant_line_fraction = (
        dominant_line_count / len(nonempty_lines) if nonempty_lines else 0.0
    )
    contains_hebrew = bool(_HEBREW.search(raw))
    hebrew_loop = (
        contains_hebrew
        and len(nonempty_lines) >= 4
        and dominant_line_fraction >= 0.5
    )
    english_fragment_loop = raw.lower().count("umably") >= 4
    try:
        ast.parse(canonical_prompt + parsed)
        syntax_valid_completion = True
    except SyntaxError:
        syntax_valid_completion = False
    return {
        "contains_hebrew": contains_hebrew,
        "hebrew_loop": hebrew_loop,
        "english_fragment_loop": english_fragment_loop,
        "collapse_loop": hebrew_loop or english_fragment_loop,
        "nonempty_line_count": len(nonempty_lines),
        "dominant_line_fraction": dominant_line_fraction,
        "syntax_valid_completion": syntax_valid_completion,
        "cap_hit": str(finish_reason).lower() == "length",
    }


def _quantile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    generated_tokens = [int(row["generated_tokens"]) for row in rows]
    finish_reasons = Counter(str(row.get("finish_reason")) for row in rows)
    parser_strategies = Counter(str(row["parser"]["strategy"]) for row in rows)
    execution_statuses = Counter(str(row["execution_status"]) for row in rows)
    passed = sum(bool(row["passed"]) for row in rows)
    return {
        "rows": len(rows),
        "passed": passed,
        "score": passed / len(rows) if rows else None,
        "contains_hebrew": sum(bool(row["classification"]["contains_hebrew"]) for row in rows),
        "hebrew_loops": sum(bool(row["classification"]["hebrew_loop"]) for row in rows),
        "english_fragment_loops": sum(
            bool(row["classification"]["english_fragment_loop"]) for row in rows
        ),
        "collapse_loops": sum(bool(row["classification"]["collapse_loop"]) for row in rows),
        "syntax_valid_completions": sum(
            bool(row["classification"]["syntax_valid_completion"]) for row in rows
        ),
        "cap_hits": sum(bool(row["classification"]["cap_hit"]) for row in rows),
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "parser_strategies": dict(sorted(parser_strategies.items())),
        "execution_statuses": dict(sorted(execution_statuses.items())),
        "execution_timeouts": sum(
            bool(row.get("execution_timed_out")) or row.get("execution_exit_code") == -24
            for row in rows
        ),
        "generated_tokens": {
            "mean": mean(generated_tokens) if generated_tokens else None,
            "p50": _quantile(generated_tokens, 0.5),
            "p90": _quantile(generated_tokens, 0.9),
            "max": max(generated_tokens) if generated_tokens else None,
        },
    }


def decide_canary_escalation(cell_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    control = cell_summaries["evalscope_chat_instruction_control"]
    candidates = [cell for cell in MATRIX_CELLS if cell != "evalscope_chat_instruction_control"]
    winner = max(
        candidates,
        key=lambda cell: (
            int(cell_summaries[cell]["passed"]),
            -int(cell_summaries[cell]["collapse_loops"]),
            int(cell_summaries[cell]["syntax_valid_completions"]),
        ),
    )
    candidate = cell_summaries[winner]
    pass_gain = int(candidate["passed"]) - int(control["passed"])
    loop_reduction = int(control["collapse_loops"]) - int(candidate["collapse_loops"])
    escalate = pass_gain >= 3 and loop_reduction >= 3
    return {
        "winning_cell": winner,
        "control_passed": int(control["passed"]),
        "candidate_passed": int(candidate["passed"]),
        "pass_gain_tasks": pass_gain,
        "control_collapse_loops": int(control["collapse_loops"]),
        "candidate_collapse_loops": int(candidate["collapse_loops"]),
        "collapse_loop_reduction_tasks": loop_reduction,
        "required_pass_gain_tasks": 3,
        "required_collapse_loop_reduction_tasks": 3,
        "escalate_to_full_untuned_base": escalate,
    }
