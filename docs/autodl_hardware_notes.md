# AutoDL and multi-GPU notes

The paper gives the Qwen3 GRPO hyperparameters, but this scaffold does not
assume a published wall-clock or hardware target. Treat 4x5090 or H800 as
adaptation targets that need pilot throughput measurements, not guaranteed
drop-in equivalents.

## 4x RTX 5090 planning

- Start with `configs/layer10_grpo.yaml`.
- Use `scripts/launch_single_node_4gpu.sh` to render the plan.
- Expect to reduce per-device rollout or micro batch sizes if memory pressure
  appears.
- Keep `max_response_length=3072` for paper alignment unless doing a clearly
  labeled pilot.
- Save logs and metrics immediately after each run.

## H800 planning

- Use bf16 when available.
- Prefer the exact same effective global batch and group size.
- If changing tensor/data parallel settings, encode that in the run name.

## Storage discipline

Default pullback after a real run:

- logs
- metrics
- resolved config
- trainable audit
- dataset manifest
- decontam manifest
- `best.pt`
- `latest.pt`

Pull intermediate checkpoints only when explicitly needed for analysis.

Keep remote run folders aligned with local archival layout:

```text
runs/<dataset_id>/<model_impl>/<run_id>/
```

For this scaffold's first pass, `scripts/remote/autodl_remote_launch.sh` writes
under `runs/<run_id>/` and `logs/<run_id>/`; once the full veRL runner is wired,
map those into the dataset/model/run layout before long benchmark waves.

## Shutdown discipline

After pulling artifacts:

1. Check no training screen is running.
2. Check GPU utilization.
3. Sync.
4. Shut down through the instance OS or provider dashboard.
