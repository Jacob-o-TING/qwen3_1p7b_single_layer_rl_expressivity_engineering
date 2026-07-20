#!/usr/bin/env bash
set -euo pipefail
export NPROC_PER_NODE=4
CONFIG="${1:-configs/layer10_grpo.yaml}"
bash "$(dirname "${BASH_SOURCE[0]}")/launch_verl_grpo.sh" "$CONFIG"
