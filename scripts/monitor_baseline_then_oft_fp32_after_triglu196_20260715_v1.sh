#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAVE=baseline_then_oft_fp32_6x5090_grpo_after_triglu196_20260715_v1
R="$ROOT/runs/grpo_reordered/$WAVE"
SOURCE="$ROOT/runs/grpo_interleaved/triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1"
PRIORITY="$ROOT/runs/grpo_priority/triglu_priority_6x5090_grpo_step128_to294_then_baseline128_to196_20260715_v1"
OLD="$ROOT/runs/grpo_serial/triglu_baseline_6x5090_grpo_20to98_serial_20260712_v1"
PY="$ROOT/envs/vllm0102_verl061/bin/python"

echo "=== Qwen GRPO: TriGLU-196 -> baseline-196 -> FP32 SwiGLU-only OFT-196 ==="
if [[ ! -f "$R/state.env" ]]; then
  phase=NOT_ARMED
  [[ -f "$R/handoff_state.env" ]] && phase=$(sed -n 's/^HANDOFF_PHASE=//p' "$R/handoff_state.env" | tail -1)
  echo "handoff: $phase"
  echo "The boundary guard is protecting TriGLU-196; successor controller has not started yet."
  echo
  bash "$ROOT/scripts/monitor_triglu_priority_to294_then_baseline196_20260715_v1.sh"
  exit 0
fi

source "$R/state.env"
printf 'phase: %s  active variant: %s  target: %s\n' "$PHASE" "$VARIANT" "$TARGET"
log=$(find "$R/logs" -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)

"$PY" - "$R" "$SOURCE" "$PHASE" "$VARIANT" "$TARGET" "${log:-}" "$START_UNIX" <<'PY'
import re
import sys
from pathlib import Path

root, source = Path(sys.argv[1]), Path(sys.argv[2])
phase, active, target = sys.argv[3], sys.argv[4], int(sys.argv[5])
log_arg, started = sys.argv[6], int(sys.argv[7])
ansi = re.compile(r"\x1b\[[0-9;]*m")

def tracker(variant):
    if variant == "baseline":
        path = source / "baseline_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1/checkpoints/latest_checkpointed_iteration.txt"
    else:
        path = root / "oft_fp32_swigluonly_6x5090_grpo_untunedbase_to196_seed20260707_v1/checkpoints/latest_checkpointed_iteration.txt"
    if not path.exists(): return 0
    return int(re.sub(r"\D", "", path.read_text(encoding="utf-8")) or 0)

segment = metrics_line = ""; live_step = progress_step = 0
if log_arg and Path(log_arg).exists():
    text = ansi.sub("", Path(log_arg).read_text(encoding="utf-8", errors="replace")).replace("\r", "\n")
    segment = text.rsplit("TRAIN_SEGMENT_START", 1)[-1]
    matches = list(re.finditer(r"(?:^|\n).*?step:(\d+) - .*", segment))
    if matches:
        live_step = int(matches[-1].group(1)); metrics_line = matches[-1].group(0).strip()
    progress = re.findall(r"Training Progress:.*?\|\s*(\d+)/(\d+)", segment)
    if progress: progress_step = int(progress[-1][0])

for path in sorted(Path("/tmp/ray/session_latest/logs").glob("worker-*.out"), key=lambda p:p.stat().st_mtime, reverse=True)[:20]:
    if path.stat().st_mtime < started - 120: continue
    with path.open("rb") as handle:
        size=handle.seek(0,2); handle.seek(max(0,size-(8<<20)))
        raw=ansi.sub("",handle.read().decode("utf-8",errors="replace")).replace("\r","\n")
    matches=list(re.finditer(r"(?:^|\n).*?step:(\d+) - .*",raw))
    if matches and int(matches[-1].group(1))>live_step:
        live_step=int(matches[-1].group(1)); metrics_line=matches[-1].group(0).strip()

steps={"baseline":tracker("baseline"),"oft":tracker("oft")}
if phase=="TRAIN" and active in steps: steps[active]=max(steps[active],live_step,progress_step)
print(f"{'triglu':8s} [{'#'*40}] 196/196 archived complete")
for name in ("baseline","oft"):
    width=40; step=steps[name]; fill=min(width, max(0,step)*width//196)
    marker=" live" if phase=="TRAIN" and active==name else ""
    print(f"{name:8s} [{'#'*fill}{'.'*(width-fill)}] {step:3d}/196{marker}")

if phase=="TRAIN" and active in steps:
    print(f"current segment: {active} step {steps[active]}/{target}; metrics through step {live_step}")
elif phase=="EVAL": print(f"current six-GPU parallel evaluation: {active} step {target}")
elif phase=="OFT_EXPORT": print("preparing exact-identity FP32 OFT model export")
elif phase=="OFT_PREFLIGHT": print("running bounded HF/vLLM OFT parity and dispatch preflight")

if metrics_line:
    fields=dict(re.findall(r"(?:^| - )([^:]+):([^ ]+)",metrics_line))
    def num(key):
        try:return float(fields[key])
        except (KeyError,ValueError):return None
    reward,kl,clip=num("critic/score/mean"),num("actor/ppo_kl"),num("actor/pg_clipfrac")
    length,cap=num("response_length/mean"),num("response_length/clip_ratio")
    step_s,rollout_s,update_s=num("timing_s/step"),num("timing_s/generate_sequences"),num("timing_s/update_actor")
    lr=num("actor/lr")
    print(f"latest metrics (step {live_step}):")
    if None not in (reward,kl,clip): print(f"  reward={reward:.4f}  ppo_kl={kl:.6f}  clipfrac={clip:.4f}")
    if lr is not None: print(f"  learning_rate={lr:.8g}")
    if None not in (length,cap): print(f"  response_mean={length:.1f} tok  cap_hit={cap:.2%}")
    if step_s and phase=="TRAIN":
        remaining=max(0,target-steps[active])
        print(f"recent speed: {step_s/60:.2f} min/update ({3600/step_s:.3f} updates/hour)")
        if None not in (rollout_s,update_s): print(f"  rollout={rollout_s/60:.2f} min  actor_update={update_s/60:.2f} min")
        print(f"  segment ETA={remaining*step_s/3600:.2f} h ({remaining} updates remaining)")

offset=segment.rfind(metrics_line) if metrics_line else -1
tail=segment[offset+len(metrics_line):] if offset>=0 else segment
errors=re.findall(r"Traceback \(most recent call last\):|OutOfMemory(?:Error)?|RayTaskError|WAVE_FAILED",tail)
if errors: print(f"ALERT: current-attempt error after latest metric ({errors[-1]})")
PY

echo "milestone evaluation comparison by global step (partial included):"
view=$(mktemp -d /tmp/qwen_unified_eval_view.XXXXXX)
trap 'rm -rf "$view"' EXIT
for step in 98 128 158 196; do
  echo "--- global step ${step} ---"
  for variant in triglu baseline oft; do
    found=""
    for root in "$R" "$PRIORITY" "$SOURCE" "$OLD"; do
      candidate="$root/evaluations/${variant}_step_${step}"
      if [[ -d "$candidate" ]]; then found="$candidate"; break; fi
    done
    if [[ -z "$found" ]]; then
      echo "${variant}: pending"
      continue
    fi
    ln -s "$found" "$view/${variant}_step_${step}"
    echo "${variant}:"
    "$PY" "$ROOT/scripts/summarize_parallel_eval.py" "$found" 2>/dev/null || true
  done
done
"$PY" "$ROOT/scripts/summarize_parallel_eval.py" "$view" \
  --compare-subdirs --steps 98 128 158 196 2>/dev/null || true

echo "data-order receipts:"
shopt -s nullglob
receipts=("$R"/data_order/*.json)
if (( ${#receipts[@]} == 0 )); then echo "  none yet"; fi
for receipt in "${receipts[@]}"; do
  "$PY" - "$receipt" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text(encoding="utf-8"))
print(f"  step {x['global_step']}: {x['status']} same-order={x['variants_normalized_equal']} ledger={x['expected_prompt_index_ledger_sha256'][:12]}")
PY
done

echo "GPU utilization / memory:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "disk free: ${free}G"
