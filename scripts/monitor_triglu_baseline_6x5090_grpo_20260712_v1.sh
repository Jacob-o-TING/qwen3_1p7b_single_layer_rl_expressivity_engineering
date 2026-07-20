#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIORITY_WAVE=triglu_priority_6x5090_grpo_step128_to294_then_baseline128_to196_20260715_v1
PRIORITY_ROOT="$ROOT/runs/grpo_priority/$PRIORITY_WAVE"
PRIORITY_MONITOR="$ROOT/scripts/monitor_triglu_priority_to294_then_baseline196_20260715_v1.sh"
if [[ -f "$PRIORITY_ROOT/state.env" ]]; then
  exec bash "$PRIORITY_MONITOR"
fi
CONTINUATION_WAVE=triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1
CONTINUATION_ROOT="$ROOT/runs/grpo_interleaved/$CONTINUATION_WAVE"
CONTINUATION_MONITOR="$ROOT/scripts/monitor_triglu_baseline_6x5090_grpo_98to196_interleaved_20260714_v1.sh"
if [[ -f "$CONTINUATION_ROOT/state.env" ]]; then
  exec bash "$CONTINUATION_MONITOR"
fi
WAVE=triglu_baseline_6x5090_grpo_20to98_serial_20260712_v1
R="$ROOT/runs/grpo_serial/$WAVE"
echo "=== Qwen GRPO production wave ==="
if [[ -f "$R/state.env" ]]; then source "$R/state.env"; else PHASE=NOT_STARTED; VARIANT=none; TARGET=0; START_UNIX=$(date +%s); fi
printf 'phase: %s  variant: %s  target: %s\n' "$PHASE" "$VARIANT" "$TARGET"
log=$(find "$R/logs" -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
"$ROOT/envs/vllm0102_verl061/bin/python" - "$R" "$PHASE" "$VARIANT" "$TARGET" "${log:-}" "$START_UNIX" <<'PY'
import re, sys
from pathlib import Path

root, phase, active_variant, target, log_arg, start_unix = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
ansi = re.compile(r"\x1b\[[0-9;]*m")

def checkpoint_step(variant: str) -> int:
    run_id = f"{variant}_6x5090_grpo_untunedbase_b98_seed20260707_v1"
    path = root / run_id / "checkpoints/latest_checkpointed_iteration.txt"
    if not path.exists():
        return 0
    digits = re.sub(r"\D", "", path.read_text())
    return int(digits or 0)

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

# Ray writes TaskRunner metrics to its worker file before its log monitor
# forwards buffered stdout into the project tee log.
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
        raw = handle.read().decode("utf-8", errors="replace")
    raw = ansi.sub("", raw).replace("\r", "\n")
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
    fill = min(width, step * width // 98)
    marker = " live" if phase == "TRAIN" and name == active_variant else ""
    print(f"{name:8s} [{'#' * fill}{'.' * (width-fill)}] {step:3d}/98{marker}")

if phase == "TRAIN" and active_variant in steps:
    checkpoint = checkpoint_step(active_variant)
    print(f"current segment: {active_variant} step {steps[active_variant]}/{target}; latest checkpoint {checkpoint}; detailed metrics through step {live_step}")

if metrics_line:
    fields = dict(re.findall(r"(?:^| - )([^:]+):([^ ]+)", metrics_line))
    def number(key: str):
        try: return float(fields[key])
        except (KeyError, ValueError): return None
    score, kl, clip = number("critic/score/mean"), number("actor/ppo_kl"), number("actor/pg_clipfrac")
    length, cap = number("response_length/mean"), number("response_length/clip_ratio")
    step_s, rollout_s, update_s = number("timing_s/step"), number("timing_s/generate_sequences"), number("timing_s/update_actor")
    throughput = number("perf/throughput")
    print(f"latest emitted metrics (step {live_step}; checkpoint writer may be ahead):")
    print(f"  reward={score:.4f}  ppo_kl={kl:.6f}  clipfrac={clip:.4f}" if None not in (score, kl, clip) else "  reward/KL metrics pending")
    print(f"  response_mean={length:.1f} tok  cap_hit={cap:.2%}" if None not in (length, cap) else "  response metrics pending")
    if step_s is not None:
        effective_step = steps.get(active_variant, live_step)
        remaining = max(0, target - effective_step)
        eta = remaining * step_s
        print(f"recent speed: {step_s/60:.2f} min/update ({3600/step_s:.3f} updates/hour)")
        print(f"  step_time={step_s/60:.2f} min  rollout={rollout_s/60:.2f} min  actor_update={update_s/60:.2f} min  throughput={throughput:.1f}")
        print(f"  segment ETA={eta/3600:.2f} h ({remaining} updates remaining)")

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
echo "evaluation results (partial or final):"
for d in "$R"/evaluations/*; do
  [[ -d "$d" ]] || continue
  echo "$(basename "$d")"
  "$ROOT/envs/vllm0102_verl061/bin/python" "$ROOT/scripts/summarize_parallel_eval.py" "$d" 2>/dev/null || true
done
"$ROOT/envs/vllm0102_verl061/bin/python" \
  "$ROOT/scripts/summarize_parallel_eval.py" "$R/evaluations" --compare-subdirs 2>/dev/null || true
echo "GPU utilization / memory:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'); echo "disk free: ${free}G"
