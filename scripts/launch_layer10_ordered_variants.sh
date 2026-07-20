#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIGS=(
  "configs/layer10_whole_layer_shs.yaml"
  "configs/layer10_whole_layer_baseline.yaml"
  "configs/layer10_whole_layer_triglu_side_ffn.yaml"
  "configs/layer10_whole_layer_oft.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "=== Launching $config ==="
  bash scripts/launch_verl_grpo.sh "$config"
done
