# AutoDL sync and remote launch

## Current persistent SSH alias

The current workstation uses this local `~/.ssh/config` alias:

```sshconfig
Host autodl-qwen
    HostName connect.example.seetacloud.com
    Port 12345
    User root
    IdentityFile ~/.ssh/id_ed25519_autodl_codex
    IdentitiesOnly yes
    AddressFamily inet
    ConnectTimeout 60
    ServerAliveInterval 20
    ServerAliveCountMax 6
    TCPKeepAlive yes
```

Connect with `ssh autodl-qwen`. AutoDL host ports are ephemeral; update the
alias when an instance is recreated or its published SSH port changes.

SeetaCloud's gateway may occasionally time out before sending the SSH server
banner (`banner exchange ... timed out`). This happens before authentication
and is not evidence of an invalid key; an invalid key normally reaches
authentication and reports `Permission denied (publickey)`. Observed SCP speed
of roughly 9-18 KB/s is consistent with the same gateway instability.

Prefer one persistent interactive SSH connection for control commands and let
remote `screen` own long-running training. Avoid repeated short SSH/SCP
handshakes where possible. One banner timeout is a normal transport issue, not
a training abnormality and never a shutdown condition by itself.

No secrets, API keys, hostnames, or private credentials belong in this repo.
Pass ephemeral AutoDL host, port, and key path as command arguments or
environment variables.

## Package from Windows

From the scaffold folder:

```powershell
.\scripts\remote\package_for_autodl.ps1
```

This writes `qwen3_1p7b_single_layer_rl_autodl.tar` and excludes local `runs`,
`data`, `checkpoints`, caches, and heavyweight artifacts.

## Sync to AutoDL

Copy the fresh SSH host and port from the provider dashboard, then:

```powershell
.\scripts\remote\sync_to_autodl.ps1 `
  -HostName connect.example.seetacloud.com `
  -Port 12345 `
  -KeyPath "$env:USERPROFILE\.ssh\id_ed25519_autodl_codex"
```

The default remote path is:

```text
/root/autodl-tmp/qwen3_1p7b_single_layer_rl
```

Use `/root/autodl-tmp` or the provider-recommended persistent project path.
Do not assume the container root filesystem is archival storage.

## Interpreter caution

Non-interactive SSH often does not load Conda on `PATH`. Use an absolute
interpreter path:

```bash
/root/miniconda3/bin/python
```

Override when needed:

```bash
PYTHON_BIN=/root/miniconda3/bin/python \
PROJECT_DIR=/root/autodl-tmp/qwen3_1p7b_single_layer_rl \
bash scripts/remote/autodl_remote_launch.sh configs/training_modes/selected_layer_no_adapter.yaml
```

## Detached launch with screen

On the remote machine:

```bash
cd /root/autodl-tmp/qwen3_1p7b_single_layer_rl
bash scripts/remote/autodl_remote_launch.sh configs/training_modes/selected_layer_no_adapter.yaml layer10_no_adapter_pilot
```

Reattach:

```bash
screen -r qwen3_rl_layer10_no_adapter_pilot
```

Inspect logs:

```bash
tail -n 200 logs/layer10_no_adapter_pilot/train.log
```

The launch script records:

- `logs/<run_id>/launch_env.txt`
- `logs/<run_id>/screen_command.sh`
- `logs/<run_id>/train.log`
- `runs/<run_id>/plan/dry_run_manifest.json`
- `runs/<run_id>/plan/resolved_config.json`

## Artifact pullback

From Windows:

```powershell
.\scripts\remote\pull_artifacts.ps1 `
  -HostName connect.example.seetacloud.com `
  -Port 12345 `
  -RunId layer10_no_adapter_pilot `
  -KeyPath "$env:USERPROFILE\.ssh\id_ed25519_autodl_codex"
```

Local layout:

```text
remote_artifacts/
  <run_id>/
    <run_id>_records.tar.gz
```

Default pullback should include logs, metrics, manifests, `best.pt`, and
`latest.pt` once real training is wired in. Pull intermediate `step_*.pt`
checkpoints only when the analysis requires them.

### Remote plot generation and pullback

Plots describing a remote run must be generated on that instance from its
authoritative logs or metrics and written under the run's own `analysis/`
directory. A durable plot bundle contains the `PNG`, the exact plotted points
as `CSV` or `JSON`, and a compact manifest with source paths, first/last step,
point count, missing-step audit, generation timestamp, and SHA-256 values.

Pull the bundle into the identical local run-relative path immediately. Verify
the local data and image hashes against the remote manifest and visually inspect
the image before presenting it. A conversation-only or locally reconstructed
chart is optional presentation, not the experiment artifact of record.

## Remote smoke test

Before a real GRPO launch:

```bash
cd /root/autodl-tmp/qwen3_1p7b_single_layer_rl
export PYTHONPATH="$PWD/src"
/root/miniconda3/bin/python -m qwen_single_layer_rl.training.dry_run \
  --config configs/smoke_tiny.yaml \
  --out runs/remote_smoke
```

If this fails, fix Python path or package sync before starting GPU work.

## Dataset download on AutoDL

Install the lightweight scaffold dependencies first:

```bash
cd /root/autodl-tmp/qwen3_1p7b_single_layer_rl
/root/miniconda3/bin/python -m pip install -r requirements.txt
```

Then download and prepare NuminaMath-CoT:

```bash
bash scripts/prepare_numina_math.sh \
  --source AI-MO/NuminaMath-CoT \
  --out data/numina_math_cot_50k \
  --target-size 50000 \
  --seed 20260707
```

The script streams the HF dataset by default. For a tiny connectivity test:

```bash
bash scripts/prepare_numina_math.sh \
  --source AI-MO/NuminaMath-CoT \
  --out data/numina_math_cot_smoke \
  --target-size 100 \
  --max-source-records 1000
```

Run `bash scripts/prepare_paper_benchmarks.sh`, then pass its
`--benchmark-problems data/decontam/qwen_math_eval_<revision>/benchmark_problems.jsonl`
output to `prepare_numina_math.sh`. This enables both normalized exact filtering
and the pinned token 8-gram near-duplicate policy; the manifest records exact and
near-match removal counts separately.
