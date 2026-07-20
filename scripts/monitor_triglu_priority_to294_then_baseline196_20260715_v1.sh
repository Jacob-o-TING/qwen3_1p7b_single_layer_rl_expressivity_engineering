#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAVE=triglu_priority_6x5090_grpo_step128_to294_then_baseline128_to196_20260715_v1
R="$ROOT/runs/grpo_priority/$WAVE"
SOURCE_ROOT="$ROOT/runs/grpo_interleaved/triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1"
REORDER_ROOT="$ROOT/runs/grpo_reordered/baseline_then_oft_fp32_6x5090_grpo_after_triglu196_20260715_v1"
OLD_ROOT="$ROOT/runs/grpo_serial/triglu_baseline_6x5090_grpo_20to98_serial_20260712_v1"
PY="$ROOT/envs/vllm0102_verl061/bin/python"

echo "=== Qwen GRPO matched continuation: both variants to 294 ==="
echo "order: baseline 196 + eval -> TriGLU 226/256/294 -> baseline 226/256/294"
echo "KL reference: own frozen step 98 through step 196; own frozen step 196 for steps 197-294"
if screen -ls 2>/dev/null | grep -q '[.]qwen_triglu_priority_to294_then_baseline196_20260715_v1[[:space:]]'; then
  echo "controller screen: running"
else
  echo "controller screen: stopped"
fi
if [[ -f "$R/WAVE_FAILED" ]]; then echo "ALERT: $(cat "$R/WAVE_FAILED")"; fi
if [[ -f "$R/HANDOFF_FAILED" ]]; then echo "ALERT: $(cat "$R/HANDOFF_FAILED")"; fi
if [[ -f "$R/handoff_state.env" ]]; then
  # shellcheck disable=SC1090
  source "$R/handoff_state.env"
  if [[ -f "$R/state.env" && "${HANDOFF_PHASE:-unknown}" == "FAILED" ]]; then
    echo "handoff history: failed attempt superseded by active successor"
  else
    echo "handoff: ${HANDOFF_PHASE:-unknown}"
  fi
fi
if [[ ! -f "$R/state.env" ]]; then
  if [[ -f "$R/handoff_state.env" ]]; then
    source "$R/handoff_state.env"
  else
    HANDOFF_PHASE=NOT_ARMED
  fi
  echo "handoff: $HANDOFF_PHASE; successor controller not active yet"
  echo
  bash "$ROOT/scripts/monitor_triglu_baseline_6x5090_grpo_20260712_v1.sh"
  exit 0
fi

source "$R/state.env"
printf 'continuation phase: %s  variant: %s  target: %s\n' "$PHASE" "$VARIANT" "$TARGET"
if [[ "$PHASE" == "TRAIN" && "$TARGET" -gt 196 ]]; then
  echo "active optimizer schedule: cosine 5e-7 at global step 196 -> 5e-8 at global step 294 (segment target: $TARGET)"
elif [[ "$PHASE" == "TRAIN" ]]; then
  echo "active optimizer schedule: cosine 5e-6 at global step 128 -> 5e-7 at global step 196"
fi
log=$(find "$R/logs" -type f -name "${VARIANT}_to_${TARGET}.log" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)

"$PY" - "$R" "$SOURCE_ROOT" "$REORDER_ROOT" "$PHASE" "$VARIANT" "$TARGET" "${log:-}" "$START_UNIX" <<'PY'
import re
import sys
from pathlib import Path

root, source, reorder = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
phase, active_variant, target = sys.argv[4], sys.argv[5], int(sys.argv[6])
log_arg, start_unix = sys.argv[7], int(sys.argv[8])
ansi = re.compile(r"\x1b\[[0-9;]*m")

def checkpoint_step(variant: str) -> int:
    third = root / f"{variant}_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1"
    tracker = third / "checkpoints" / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        tracker = source / f"{variant}_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1" / "checkpoints" / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return 128
    digits = re.sub(r"\D", "", tracker.read_text(encoding="utf-8"))
    return int(digits or 98)

segment = ""
metrics_line = ""
metrics_step = 0
metrics_source = "current attempt"
live_step = 0
progress_step = 0
if log_arg and Path(log_arg).exists():
    text = ansi.sub("", Path(log_arg).read_text(encoding="utf-8", errors="replace")).replace("\r", "\n")
    segment = text.rsplit("TRAIN_SEGMENT_START", 1)[-1]
    matches = list(re.finditer(r"(?:^|\n).*?step:(\d+) - .*", segment))
    if matches:
        live_step = int(matches[-1].group(1))
        metrics_step = live_step
        metrics_line = matches[-1].group(0).strip()
    progress = re.findall(r"Training Progress:.*?\|\s*(\d+)/(\d+)", segment)
    if progress:
        progress_step = int(progress[-1][0])

ray_logs = sorted(
    (
        path for path in Path("/tmp/ray/session_latest/logs").glob("worker-*.out")
        if path.stat().st_mtime >= start_unix - 120
    ),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
for path in ray_logs[:20]:
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size - (8 << 20)))
        raw = ansi.sub("", handle.read().decode("utf-8", errors="replace")).replace("\r", "\n")
    matches = list(re.finditer(r"(?:^|\n).*?step:(\d+) - .*", raw))
    if matches and int(matches[-1].group(1)) > live_step:
        live_step = int(matches[-1].group(1))
        metrics_step = live_step
        metrics_line = matches[-1].group(0).strip()

if not metrics_line and active_variant in {"triglu", "baseline"}:
    fallback_logs = []
    for history_root in (root, reorder, source):
        log_root = history_root / "logs"
        fallback_logs.extend(log_root.glob(f"{active_variant}*.log"))
    for path in sorted(fallback_logs, key=lambda item: item.stat().st_mtime, reverse=True):
        raw = ansi.sub("", path.read_text(encoding="utf-8", errors="replace")).replace("\r", "\n")
        matches = list(re.finditer(r"(?:^|\n).*?step:(\d+) - .*", raw))
        if matches:
            metrics_step = int(matches[-1].group(1))
            metrics_line = matches[-1].group(0).strip()
            metrics_source = f"archived log {path.name}"
            break

steps = {name: checkpoint_step(name) for name in ("triglu", "baseline")}
if phase == "TRAIN" and active_variant in steps:
    steps[active_variant] = max(steps[active_variant], live_step, progress_step)
for name in ("triglu", "baseline"):
    step = steps[name]
    width = 40
    final = 294
    fill = min(width, max(0, step - 98) * width // (final - 98))
    marker = " live" if phase == "TRAIN" and name == active_variant else ""
    print(f"{name:8s} [{'#' * fill}{'.' * (width-fill)}] {step:3d}/{final}{marker}")

if phase == "TRAIN" and active_variant in steps:
    checkpoint = checkpoint_step(active_variant)
    print(
        f"current segment: {active_variant} step {steps[active_variant]}/{target}; "
        f"latest checkpoint {checkpoint}; metrics through step {metrics_step or 'pending'}"
    )
elif phase == "EVAL":
    pair = "baseline" if active_variant == "triglu" else "triglu"
    print(f"current evaluation: {active_variant} step {target}; paired {pair} cell may be pending")

if metrics_line:
    fields = dict(re.findall(r"(?:^| - )([^:]+):([^ ]+)", metrics_line))
    def number(key: str):
        try:
            return float(fields[key])
        except (KeyError, ValueError):
            return None
    reward = number("critic/score/mean")
    kl = number("actor/ppo_kl")
    clip = number("actor/pg_clipfrac")
    length = number("response_length/mean")
    cap = number("response_length/clip_ratio")
    step_s = number("timing_s/step")
    rollout_s = number("timing_s/generate_sequences")
    update_s = number("timing_s/update_actor")
    throughput = number("perf/throughput")
    learning_rate = number("actor/lr")
    print(f"latest completed metrics (step {metrics_step}; {metrics_source}):")
    if None not in (reward, kl, clip):
        print(f"  reward={reward:.4f}  ppo_kl={kl:.6f}  clipfrac={clip:.4f}")
    if learning_rate is not None:
        if metrics_step > 196:
            schedule = "cosine 5e-7 to 5e-8 at step 294"
        elif metrics_step > 128:
            schedule = "cosine 5e-6 to 5e-7 at step 196"
        else:
            schedule = "constant 5e-6"
        print(f"  learning_rate={learning_rate:.8g}  metric_schedule={schedule}")
    if None not in (length, cap):
        print(f"  response_mean={length:.1f} tok  cap_hit={cap:.2%}")
    if step_s is not None and phase == "TRAIN":
        effective = steps.get(active_variant, live_step)
        remaining = max(0, target - effective)
        print(f"recent speed: {step_s/60:.2f} min/update ({3600/step_s:.3f} updates/hour)")
        if None not in (rollout_s, update_s, throughput):
            print(
                f"  step_time={step_s/60:.2f} min  rollout={rollout_s/60:.2f} min  "
                f"actor_update={update_s/60:.2f} min  throughput={throughput:.1f}"
            )
        print(f"  segment ETA={remaining * step_s / 3600:.2f} h ({remaining} updates remaining)")

metric_offset = segment.rfind(metrics_line) if metrics_line else -1
tail = segment[metric_offset + len(metrics_line):] if metric_offset >= 0 else segment
errors = re.findall(
    r"(?m)(Traceback \(most recent call last\):|Error executing job with overrides:|"
    r"OutOfMemory(?:Error)?|RayTaskError|^.*KeyError:|^RUN_EXIT [1-9][0-9]*)",
    tail,
)
if errors and phase in {"TRAIN", "EVAL"}:
    print(f"ALERT: current-attempt error after latest completed step ({errors[-1]})")
PY

echo
echo "=== Core math evaluation results ==="
echo "milestone evaluation comparison by global step (partial included):"
view=$(mktemp -d /tmp/qwen_triglu_priority_eval_view.XXXXXX)
trap 'rm -rf "$view"' EXIT
for step in 98 128 158 196 226 256 294; do
  echo "--- global step ${step} ---"
  for variant in triglu baseline; do
    directory=""
    for root in "$R" "$REORDER_ROOT" "$SOURCE_ROOT" "$OLD_ROOT"; do
      candidate="$root/evaluations/${variant}_step_${step}"
      if [[ -d "$candidate" ]]; then directory="$candidate"; break; fi
    done
    if [[ -z "$directory" ]]; then
      echo "${variant}: pending"
      continue
    fi
    ln -s "$directory" "$view/${variant}_step_${step}"
    echo "${variant}:"
    "$PY" "$ROOT/scripts/summarize_parallel_eval.py" "$directory" 2>/dev/null || true
  done
done
"$PY" "$ROOT/scripts/summarize_parallel_eval.py" "$view" \
  --compare-subdirs --steps 98 128 158 196 226 256 294 2>/dev/null || true

echo
echo "=== OOD / out-of-domain evaluation ==="
"$PY" "$ROOT/scripts/summarize_qwen_eval_dashboard.py" "$ROOT" 2>/dev/null || true
echo

if [[ "$PHASE" == "WAITING_PRE_BASELINE_OOD" ]]; then
  OOD_BASE="$ROOT/runs/ood_eval/qwen3_1p7b_ood_6x5090_step294_20260716_v1"
  B196="$ROOT/runs/ood_eval/qwen3_1p7b_ood_6x5090_baseline_step196_20260717_v1"
  if [[ -f "$B196/state.env" && ! -f "$B196/OOD_COMPLETE" ]]; then
    current_model=baseline_step196
    OOD="$B196"
    cells=(reasoning_gpqa_staged6 reasoning_mmlupro_staged6 code_humanevalplus_staged6 code_mbpp_staged6 language_ceval_staged6 language_ifeval_staged6 language_mgsm_staged6 code_lcb)
  else
    current_model=$(awk -F= '$1 == "MODEL" {print $2}' "$OOD_BASE/state.env" 2>/dev/null | tail -1)
    [[ -n "$current_model" ]] || current_model=triglu
    OOD="$OOD_BASE/$current_model"
    cells=(code_heplus_mbpp code_lcb reasoning_gpqa reasoning_mmlupro language_ceval language_ifeval_mgsm)
  fi
  if screen -ls 2>/dev/null | grep -Eq '[.](qwen_ood_step294_20260716_v1|qwen_triglu_lcb6_release_latest_20260717_v1|qwen_baseline_step196_ood6_20260717_v1)[[:space:]]'; then
    echo "pre-baseline OOD: active (current model: $current_model; baseline training remains blocked)"
  else
    echo "pre-baseline OOD: watcher not running"
  fi
  triglu_status=pending
  untuned_status=pending
  baseline196_status=pending
  [[ -f "$OOD_BASE/triglu/PARALLEL_OOD_EVAL_COMPLETE" ]] && triglu_status=complete
  [[ -f "$OOD_BASE/untuned_base/PARALLEL_OOD_EVAL_COMPLETE" ]] && untuned_status=complete
  if [[ -f "$B196/PARALLEL_OOD_EVAL_COMPLETE" ]]; then
    baseline196_status=complete
  elif [[ -f "$B196/state.env" ]]; then
    baseline196_status=active
  fi
  echo "  model status: triglu=$triglu_status | untuned_base=$untuned_status | baseline_step196=$baseline196_status"
  active_cell=""
  active_progress=""
  active_mtime=0
  complete_cells=0
  for cell in "${cells[@]}"; do
    status=pending
    if [[ -f "$OOD/$cell/RANK_COMPLETE" ]]; then
      status=complete
      complete_cells=$((complete_cells + 1))
    fi
    progress=""
    cell_log="$OOD/$cell.log"
    if [[ ! -f "$cell_log" && -d "$OOD/$cell" ]]; then
      cell_log=$(find "$OOD/$cell" -maxdepth 1 -type f -name 'shard_*.log' -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2- || true)
    fi
    if [[ "$status" != complete && -n "$cell_log" && -f "$cell_log" ]]; then
      progress=$(tr '\r' '\n' <"$cell_log" \
        | grep -E 'Evaluating\[|Running\[eval\]|Downloading data:' \
        | tail -1 \
        | sed -E 's/\x1B\[[0-9;?]*[ -\/]*[@-~]//g' \
        | tail -c 180 || true)
      mtime=$(stat -c %Y "$cell_log" 2>/dev/null || echo 0)
      if (( mtime >= active_mtime )); then
        active_mtime=$mtime
        active_cell=$cell
        active_progress=$progress
      fi
    fi
    result_detail=""
    if [[ "$current_model" == baseline_step196 && -d "$OOD/$cell/shards" ]]; then
      result_detail=$("$PY" - "$OOD/$cell" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
merged = root / "merge_summary.json"
if merged.exists():
    value = json.loads(merged.read_text(encoding="utf-8"))
    print(
        f"final={100 * float(value['score']):.3f}% "
        f"n={int(value['report_samples'])}/{int(value['expected_identities'])}"
    )
else:
    cells = []
    completed_shards = 0
    for shard in sorted((root / "shards").glob("shard_*")):
        completed_shards += int((shard / "RANK_COMPLETE").exists())
        reports = []
        for path in shard.glob("main/*/reports/*/*.json"):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                reports.append((path, int(report["num"]), float(report["score"])))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        if reports:
            cells.append(sorted(reports, key=lambda item: str(item[0]))[-1])
    samples = sum(item[1] for item in cells)
    if samples:
        score = sum(item[1] * item[2] for item in cells) / samples
        print(
            f"shards={completed_shards}/6 partial={100 * score:.3f}% "
            f"n={samples} (not final until exact merge)"
        )
PY
)
      if [[ "$status" == pending && -n "$result_detail" ]]; then
        status=partial
      fi
    fi
    printf '  %-24s %-8s %s %s\n' "$cell" "$status" "$result_detail" "$progress"
  done
  echo "  model cells complete: $complete_cells/${#cells[@]}"
  if [[ -n "$active_cell" ]]; then
    echo "  current eval: $current_model / $active_cell"
    [[ -n "$active_progress" ]] && echo "  current progress (latest active shard): $active_progress"
  fi
  LCB_PARALLEL="$OOD/code_lcb_parallel6_release_latest_20260717_v1"
  if [[ -d "$LCB_PARALLEL/shards" ]]; then
    echo "  $current_model LiveCodeBench release_latest six-way shards:"
    for shard in 00 01 02 03 04 05; do
      shard_dir="$LCB_PARALLEL/shards/shard_$shard"
      status=pending
      [[ -f "$shard_dir/RANK_COMPLETE" ]] && status=complete
      generated=0
      if [[ -f "$shard_dir/generation_receipts.jsonl" ]]; then
        generated=$(grep -c '"event": "generation_completed"' \
          "$shard_dir/generation_receipts.jsonl" 2>/dev/null || true)
      fi
      printf '    shard_%s %-8s generated=%s\n' "$shard" "$status" "$generated"
    done
    if [[ -f "$LCB_PARALLEL/merge_summary.json" ]]; then
      "$PY" - "$LCB_PARALLEL/merge_summary.json" "$current_model" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if sys.argv[2] == "triglu":
    print(
        f"    historical merged report={100 * summary['score']:.3f}% "
        f"INVALID (sandbox output-contract artifact) "
        f"samples={summary['samples']} unique={summary['identity_unique']}"
    )
else:
    print(
        f"    merged score={100 * summary['score']:.3f}% "
        f"samples={summary['samples']} unique={summary['identity_unique']}"
    )
PY
    fi
  fi
  for summary_model in triglu untuned_base baseline; do
    summary_root="$OOD_BASE/$summary_model"
    [[ -f "$summary_root/PARALLEL_OOD_EVAL_COMPLETE" ]] || continue
    echo "  $summary_model OOD scores (corrected evaluator evidence preferred):"
    "$PY" "$ROOT/scripts/summarize_ood_eval.py" "$summary_root" 2>/dev/null || true
  done
  if [[ -d "$B196" ]]; then
    echo "  baseline_step196 OOD scores (same presentation; partial included):"
    "$PY" "$ROOT/scripts/summarize_ood_eval.py" "$B196" 2>/dev/null || true
  fi
fi

bash "$ROOT/scripts/monitor_qwen3_1p7b_ood_6x5090_step294_20260716_v1.sh" --embedded

MAJOR_OTHER="$ROOT/runs/ood_eval/qwen3_1p7b_other_eval_6x5090_triglu_baseline_steps158_196_226_256_294_20260718_v1"
if [[ -d "$MAJOR_OTHER" ]]; then
  echo
  bash "$ROOT/scripts/monitor_qwen3_1p7b_other_eval_majorsteps_6x5090_20260718_v1.sh" --embedded
fi

echo "paired data-order receipts:"
for receipt in "$R"/data_order/*.json; do
  [[ -f "$receipt" ]] || continue
  "$PY" - "$receipt" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
print(
    f"  {path.stem}: {payload['status']} normalized_equal={payload['variants_normalized_equal']} "
    f"ledger={payload['expected_prompt_index_ledger_sha256'][:12]}"
)
PY
done

echo "GPU utilization / memory:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "disk free: ${free}G"
