#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/training_modes/selected_layer_no_adapter.yaml}"
RUN_ID="${2:-$(basename "$CONFIG" .yaml)_$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SCREEN_NAME="${SCREEN_NAME:-qwen3_rl_${RUN_ID}}"
LOG_DIR="${PROJECT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/train.log"

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR" "runs/$RUN_ID"

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required for detached AutoDL launch." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN explicitly, for example /root/miniconda3/bin/python." >&2
  exit 1
fi

cat > "${LOG_DIR}/launch_env.txt" <<MSG
PROJECT_DIR=$PROJECT_DIR
PYTHON_BIN=$PYTHON_BIN
CONFIG=$CONFIG
RUN_ID=$RUN_ID
SCREEN_NAME=$SCREEN_NAME
LOG_FILE=$LOG_FILE
MSG

cat > "${LOG_DIR}/screen_command.sh" <<'SCRIPT'
set -euo pipefail
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
set -o pipefail
{
  echo "RUN_START $(date -Is)"
  echo "HOST $(hostname)"
  echo "PYTHON_BIN $PYTHON_BIN"
  "$PYTHON_BIN" --version
  nvidia-smi || true
  "$PYTHON_BIN" -m qwen_single_layer_rl.training.dry_run --config "$CONFIG" --out "runs/$RUN_ID/plan"
  bash scripts/launch_verl_grpo.sh "$CONFIG"
  status=$?
  echo "RUN_EXIT $status"
  echo "RUN_END $(date -Is)"
  exit "$status"
} 2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
SCRIPT

chmod +x "${LOG_DIR}/screen_command.sh"

screen -dmS "$SCREEN_NAME" bash -lc \
  "PROJECT_DIR='$PROJECT_DIR' PYTHON_BIN='$PYTHON_BIN' CONFIG='$CONFIG' RUN_ID='$RUN_ID' LOG_FILE='$LOG_FILE' bash '${LOG_DIR}/screen_command.sh'"

echo "Launched screen: $SCREEN_NAME"
echo "Log: $LOG_FILE"
echo "Reattach: screen -r $SCREEN_NAME"
