#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAVE=triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1
R="$ROOT/runs/grpo_interleaved/$WAVE"
OLD_ROOT="$ROOT/runs/grpo_serial/triglu_baseline_6x5090_grpo_20to98_serial_20260712_v1"
PY="$ROOT/envs/vllm0102_verl061/bin/python"

echo "=== Qwen GRPO interleaved continuation ==="
echo "Next-98 milestones: global 128=round2+30, 158=round2+60, 196=round2+98"
if [[ -f "$R/autostart_state.env" ]]; then
  source "$R/autostart_state.env"
else
  AUTOSTART_PHASE=NOT_ARMED
  OLD_STEP=0
  UPDATED_UNIX=0
fi
printf 'autostart: %s  old baseline: %s/98\n' "$AUTOSTART_PHASE" "$OLD_STEP"

if [[ "$AUTOSTART_PHASE" == FAILED && -f "$R/AUTOSTART_FAILED" ]]; then
  echo "ALERT: $(cat "$R/AUTOSTART_FAILED")"
fi
if [[ -f "$R/WAVE_FAILED" ]]; then
  echo "ALERT: $(cat "$R/WAVE_FAILED")"
fi

if [[ ! -f "$R/state.env" ]]; then
  echo "continuation: waiting for the old step-98 training and evaluation wave"
  echo
  bash "$ROOT/scripts/monitor_triglu_baseline_6x5090_grpo_20260712_v1.sh"
  exit 0
fi

source "$R/state.env"
printf 'continuation phase: %s  variant: %s  target: %s\n' "$PHASE" "$VARIANT" "$TARGET"
log=$(find "$R/logs" -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)

"$PY" - "$R" "$PHASE" "$VARIANT" "$TARGET" "${log:-}" "$START_UNIX" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
phase, active_variant, target = sys.argv[2], sys.argv[3], int(sys.argv[4])
log_arg, start_unix = sys.argv[5], int(sys.argv[6])
ansi = re.compile(r"\x1b\[[0-9;]*m")

def checkpoint_step(variant: str) -> int:
    run_id = f"{variant}_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1"
    tracker = root / run_id / "checkpoints" / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return 98
    digits = re.sub(r"\D", "", tracker.read_text(encoding="utf-8"))
    return int(digits or 98)

segment = ""
metrics_line = ""
live_step = 0
progress_step = 0
if log_arg and Path(log_arg).exists():
    text = ansi.sub("", Path(log_arg).read_text(encoding="utf-8", errors="replace")).replace("\r", "\n")
    segment = text.rsplit("TRAIN_SEGMENT_START", 1)[-1]
    matches = list(re.finditer(r"(?:^|\n).*?step:(\d+) - .*", segment))
    if matches:
        live_step = int(matches[-1].group(1))
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
        metrics_line = matches[-1].group(0).strip()

steps = {name: checkpoint_step(name) for name in ("triglu", "baseline")}
if phase == "TRAIN" and active_variant in steps:
    steps[active_variant] = max(steps[active_variant], live_step, progress_step)
for name in ("triglu", "baseline"):
    step = steps[name]
    width = 40
    fill = min(width, max(0, step - 98) * width // 98)
    marker = " live" if phase == "TRAIN" and name == active_variant else ""
    print(f"{name:8s} [{'#' * fill}{'.' * (width-fill)}] {step:3d}/196{marker}")

if phase == "TRAIN" and active_variant in steps:
    checkpoint = checkpoint_step(active_variant)
    print(
        f"current segment: {active_variant} step {steps[active_variant]}/{target}; "
        f"latest checkpoint {checkpoint}; metrics through step {live_step}"
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
    print(f"latest metrics (step {live_step}):")
    if None not in (reward, kl, clip):
        print(f"  reward={reward:.4f}  ppo_kl={kl:.6f}  clipfrac={clip:.4f}")
    if learning_rate is not None:
        schedule = "constant" if live_step <= 128 else "cosine decay to 5e-7 at step 196"
        print(f"  learning_rate={learning_rate:.8g}  schedule={schedule}")
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
if errors:
    print(f"ALERT: current-attempt error after latest completed step ({errors[-1]})")
PY

echo "Next-98 evaluation results (partial or final; historical global 30/60 excluded):"
for step in 128 158 196; do
  for variant in triglu baseline; do
    directory="$R/evaluations/${variant}_step_${step}"
    [[ -d "$directory" ]] || continue
    echo "$(basename "$directory")"
    "$PY" "$ROOT/scripts/summarize_parallel_eval.py" "$directory" 2>/dev/null || true
  done
done
"$PY" "$ROOT/scripts/summarize_parallel_eval.py" "$R/evaluations" \
  --compare-subdirs --steps 128 158 196 2>/dev/null || true

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
