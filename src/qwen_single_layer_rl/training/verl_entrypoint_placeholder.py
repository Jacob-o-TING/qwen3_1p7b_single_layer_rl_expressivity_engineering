from __future__ import annotations

import argparse
import json

from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.training.verl_bridge import render_verl_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(json.dumps(render_verl_plan(cfg), indent=2, sort_keys=True))
    raise SystemExit(
        "This is a placeholder. Install veRL and replace this module with the "
        "project-specific GRPO trainer entry point."
    )


if __name__ == "__main__":
    main()
