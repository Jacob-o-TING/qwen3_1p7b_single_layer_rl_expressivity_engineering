from __future__ import annotations

import argparse
import json
from pathlib import Path


CELL = "evalscope_raw_instruction_nochat"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _partial(root: Path) -> tuple[int, int]:
    reviewed = 0
    workers = 0
    for progress_path in root.glob("workers/*/progress.json"):
        progress = _read_json(progress_path)
        if progress is None:
            continue
        reviewed += int(progress.get("reviewed", 0))
        workers += 1
    return reviewed, workers


def _format_model(label: str, root: Path) -> str:
    summary = _read_json(root / "summary.json")
    if summary is not None:
        cell = summary["cells"][CELL]
        return (
            f"  {label:17s} corrected={cell['passed']}/{cell['rows']}="
            f"{100.0 * float(cell['score']):.3f}% "
            f"loops={cell['collapse_loops']} syntax={cell['syntax_valid_completions']} "
            f"cap={cell['cap_hits']} timeouts={cell.get('execution_timeouts', 'legacy-unreported')} "
            "[complete]"
        )
    reviewed, workers = _partial(root)
    state = _read_json(root / "preflight_manifest.json")
    if reviewed:
        return f"  {label:17s} corrected progress={reviewed}/164 workers_seen={workers}/6 [running]"
    if state is not None:
        return f"  {label:17s} corrected=prepared (ledger locked; generation pending)"
    return f"  {label:17s} corrected=pending"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("full_root", type=Path)
    parser.add_argument("--canary-root", type=Path)
    args = parser.parse_args()

    print("HumanEval+ prompt-protocol correction (same raw no-chat ledger):")
    if args.canary_root:
        canary = _read_json(args.canary_root / "summary.json")
        if canary is not None:
            cells = canary["cells"]
            chat = cells["evalscope_chat_instruction_control"]
            raw = cells[CELL]
            canonical = cells["canonical_completion_nochat"]
            print(
                "  canary32: "
                f"chat={chat['passed']}/32, raw-nochat={raw['passed']}/32, "
                f"canonical={canonical['passed']}/32"
            )
    print(_format_model("untuned_base", args.full_root))
    print(_format_model("triglu_step294", args.full_root / "models" / "triglu_step294"))
    print(_format_model("baseline_step196", args.full_root / "models" / "baseline_step196"))
    print("  paper untuned-base anchor=44.5% (orientation only; exact protocol parity not claimed)")


if __name__ == "__main__":
    main()
