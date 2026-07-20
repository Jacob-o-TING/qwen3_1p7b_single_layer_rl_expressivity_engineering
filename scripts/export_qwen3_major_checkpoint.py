from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.eval.gpqa_freeform import write_json_atomic


def validate_checkpoint(checkpoint: Path) -> None:
    actor = checkpoint / "actor"
    required = (checkpoint / "data.pt", actor / "fsdp_config.json")
    if any(not path.is_file() for path in required):
        raise RuntimeError(f"checkpoint metadata is incomplete: {checkpoint}")
    for prefix in ("model", "optim", "extra_state"):
        count = len(list(actor.glob(f"{prefix}_world_size_6_rank_*.pt")))
        if count != 6:
            raise RuntimeError(f"{checkpoint} has {count} {prefix} shards, expected 6")


def complete_export(path: Path) -> bool:
    return (path / "EXPORT_COMPLETE").is_file() and (path / "config.json").is_file() and bool(
        list(path.glob("*.safetensors"))
    )


def stage_triglu(actor: Path, staging: Path) -> Path:
    if staging.parent.exists():
        shutil.rmtree(staging.parent)
    staging.mkdir(parents=True)
    for source in actor.iterdir():
        if source.name == "huggingface":
            continue
        (staging / source.name).symlink_to(source)
    shutil.copytree(actor / "huggingface", staging / "huggingface")
    module = staging / "huggingface" / "triglu_hf_model.py"
    text = module.read_text(encoding="utf-8")
    needle = "from . import TRIGLU_ARCHITECTURE"
    if text.count(needle) != 1:
        raise RuntimeError(f"expected one package constant import in {module}")
    module.write_text(text.replace(needle, 'TRIGLU_ARCHITECTURE = "Qwen3TriGLUForCausalLM"'), encoding="utf-8")
    (module.parent / "configuration_qwen3_triglu.py").write_text(
        "from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUConfig\n", encoding="utf-8"
    )
    (module.parent / "modeling_qwen3_triglu.py").write_text(
        "from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUForCausalLM, Qwen3TriGLUModel\n",
        encoding="utf-8",
    )
    return staging


def finish_triglu_config(runtime_config: Path, output: Path) -> None:
    from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUConfig

    config = load_config(runtime_config)
    raw = json.loads((output / "config.json").read_text(encoding="utf-8"))
    for key in ("model_type", "architectures", "triglu_variant", "auto_map"):
        raw.pop(key, None)
    custom = Qwen3TriGLUConfig(triglu_variant=config["architecture_variant"]["params"], **raw)
    custom.save_pretrained(output)
    (output / "configuration_qwen3_triglu.py").write_text(
        "from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUConfig\n", encoding="utf-8"
    )
    (output / "modeling_qwen3_triglu.py").write_text(
        "from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUForCausalLM, Qwen3TriGLUModel\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("triglu", "baseline"), required=True)
    parser.add_argument("--global-step", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    if args.reuse and complete_export(args.reuse):
        write_json_atomic(
            args.receipt,
            {
                "status": "REUSED_EXISTING_EXPORT",
                "variant": args.variant,
                "global_step": args.global_step,
                "export_path": str(args.reuse.resolve()),
            },
        )
        print(args.reuse.resolve())
        return
    validate_checkpoint(args.checkpoint)
    if complete_export(args.output):
        write_json_atomic(
            args.receipt,
            {
                "status": "REUSED_RUN_LOCAL_EXPORT",
                "variant": args.variant,
                "global_step": args.global_step,
                "source_checkpoint": str(args.checkpoint.resolve()),
                "export_path": str(args.output.resolve()),
            },
        )
        print(args.output.resolve())
        return
    if args.output.exists():
        shutil.rmtree(args.output)
    merge_source = args.checkpoint / "actor"
    staging = None
    if args.variant == "triglu":
        staging = args.staging_root / f"triglu_step{args.global_step}" / "actor"
        merge_source = stage_triglu(merge_source, staging)
    env = dict(os.environ)
    env["HF_MODULES_CACHE"] = str((staging or args.output) / ".hf_modules_cache")
    subprocess.run(
        [
            os.sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--trust-remote-code",
            "--local_dir",
            str(merge_source),
            "--target_dir",
            str(args.output),
        ],
        check=True,
        env=env,
    )
    if args.variant == "triglu":
        finish_triglu_config(args.runtime_config, args.output)
    if not (args.output / "config.json").is_file() or not list(args.output.glob("*.safetensors")):
        raise RuntimeError(f"export is incomplete: {args.output}")
    (args.output / "EXPORT_COMPLETE").write_text("complete\n", encoding="utf-8")
    if staging and staging.parent.exists():
        shutil.rmtree(staging.parent)
    write_json_atomic(
        args.receipt,
        {
            "status": "EXPORT_COMPLETE",
            "variant": args.variant,
            "global_step": args.global_step,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "export_path": str(args.output.resolve()),
            "weight_files": [path.name for path in sorted(args.output.glob("*.safetensors"))],
        },
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
