# Qwen3-1.7B TriGLU/Baseline GRPO Matched-10 Fresh-Server Reproduction Guide

## Scope / 这份 bundle 管什么

这份 archive 是当前 Qwen3-1.7B single-layer GRPO research wave 的 offline-capable
reproduction package。它以 **TriGLU 与 whole-layer baseline 在 matched global steps
158/196/226/256/294 的 10 个 exact-resume checkpoints** 为核心，同时保存 source、configs、
tests、reports、Math/OOD evidence、manual audits、decontaminated data、original base weights、
frozen KL references 与 environment receipts。

它不是整个 workspace 的无差别镜像。旧 SFT weight tar、private chat backups、可重建 cache、
conda environment directory 和重复 exported checkpoints deliberately excluded，避免把 unrelated
历史数据和私密内容混入 scientific archive。

Canonical checkpoint map:

```text
docs/reproduction/2026-07-19_matched10_checkpoint_map.json
```

## Archive layout / 解压后的布局

Archive paths 保留 `/root/autodl-tmp` 下的相对布局：

```text
qwen3_1p7b_single_layer_rl/       # source, records, data, selected runs/evals/checkpoints
qwen3_single_layer_rl/models/Qwen3-1.7B-Base/
verl-v0.6.1-qwenpatch/
```

The archive is standard gzip and does not require `pigz` to extract. The build host used `pigz 2.6`
with 32 workers only to reduce packaging wall time; this does not alter scientific content or the
decompression format。

推荐在新机器上执行：

```bash
mkdir -p /root/autodl-tmp
tar -xzf qwen3_1p7b_triglu_baseline_grpo_matched10_full_repro_20260719_v1.tar.gz \
  -C /root/autodl-tmp
cd /root/autodl-tmp/qwen3_1p7b_single_layer_rl
(cd /root/autodl-tmp && sha256sum -c \
  qwen3_1p7b_single_layer_rl/archive_ready/qwen3_1p7b_triglu_baseline_grpo_matched10_full_repro_20260719_v1_metadata/source_sha256sums.txt)
```

Archive-level SHA-256 由同目录 `.sha256` file 给出；先验证 archive，再解压并验证 source
checksums。Receipt JSON 记录 file count、source bytes、archive bytes 与 verification status。

Finalized bundle receipt:

```text
archive: qwen3_1p7b_triglu_baseline_grpo_matched10_full_repro_20260719_v1.tar.gz
bytes:   64806065479
sha256:  d673478d66be3dca48a706636a565d7f28332dc1eb938c32c8bc85ca9e8e73ba
entries: 25333
checks:  gzip_test=pass, tar_list_test=pass
```

The archive, portable basename-only `.sha256`, external receipt JSON, source checksum list and metadata
directory live together under `archive_ready/` on the packaging host。

## Hardware and runtime contract / 环境合同

Authoritative production host at packaging time:

- 6 x NVIDIA GeForce RTX 5090, 32607 MiB each;
- NVIDIA driver `580.105.08`;
- Python `3.12.3`;
- PyTorch `2.8.0+cu128`;
- vLLM `0.10.2`;
- Triton `3.4.0`;
- Transformers `4.57.1`;
- EvalScope `1.8.1`;
- `math-verify==0.9.0`, veRL `0.6.1`, Ray `2.56.0`;
- NumPy `1.26.4`, pandas `2.2.3`, PyArrow `24.0.0`, datasets `4.8.4`,
  accelerate `1.14.0`, safetensors `0.8.0`;
- `flash_attn` was absent; vLLM used its PyTorch-native sampling fallback where recorded.

The archive carries separate complete `pip freeze` files and machine-readable receipts for the historical
training/rollout environment `envs/vllm0102_verl061` and scoring environment `envs/evalscope181`.
`/root/miniconda3` is not the production source of truth and is deliberately not used for these receipts。
Do not blindly install every unrelated utility from `pip freeze`; use it as the exact receipt. A clean
training/rollout environment can be built with the pinned core stack, then install the bundled project and
patched veRL:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  vllm==0.10.2 triton==3.4.0 transformers==4.57.1 \
  evalscope==1.8.1 math-verify==0.9.0 ray==2.56.0 \
  numpy==1.26.4 pandas==2.2.3 pyarrow==24.0.0 datasets==4.8.4 \
  accelerate==1.14.0 safetensors==0.8.0 pyyaml
python -m pip install -e /root/autodl-tmp/verl-v0.6.1-qwenpatch
python -m pip install -e /root/autodl-tmp/qwen3_1p7b_single_layer_rl
```

The historical EvalScope scoring environment was separate: Python `3.12.3`, Torch
`2.12.1+cu130`, Triton `3.7.1`, Transformers `4.57.1`, EvalScope `1.8.1`, Ray `2.47.1`,
NumPy `2.5.1`, pandas `3.0.3`, PyArrow `24.0.0`, datasets `4.8.4`, accelerate `1.14.0`
and safetensors `0.8.0`; it did not contain vLLM or math-verify。Use its bundled freeze when exact
historical scoring parity matters. Generation/rollout remains governed by the training/rollout environment。

CUDA/PyTorch wheels must match the new host driver. On a different GPU generation, first run the
repository's environment, BF16 optimizer, vLLM registration, checkpoint-load and weight-sync smoke
gates; do not launch production merely because imports succeed。

## Original model weights / 原始权重

Offline copy is bundled at:

```text
/root/autodl-tmp/qwen3_single_layer_rl/models/Qwen3-1.7B-Base
```

Canonical public source is `Qwen/Qwen3-1.7B-Base`. If the bundled copy is unavailable, download it with:

```bash
huggingface-cli download Qwen/Qwen3-1.7B-Base \
  --local-dir /root/autodl-tmp/qwen3_single_layer_rl/models/Qwen3-1.7B-Base
```

The packaged `model.safetensors` must have SHA-256:

```text
6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841
```

Public source: <https://huggingface.co/Qwen/Qwen3-1.7B-Base>.

## Dataset and deterministic order / 数据与顺序

The bundle carries the authoritative decontaminated materialization:

```text
data/numina_math_cot_50k_decontam_v3/
data/numina_math_cot_50k_decontam_v3_verl/
data_manifests/numina_math_cot_50k_decontam_v3/
```

Canonical upstream is `AI-MO/NuminaMath-CoT` at revision
`e8b6aad745189763d0fd9521ac0844ce44675bef`. The mirror did not expose an immutable provider
revision, so generated provenance manifests and normalized content/order digests are the actual local
integrity identity。Public source:
<https://huggingface.co/datasets/AI-MO/NuminaMath-CoT>.

Approved paired contract remains: seed `20260707`, `shuffle=true`, `group_size=4`, six-rank
`504/126/6`, identical veRL parquet hashes and identical sample order across TriGLU/baseline。

## Exact resume and KL references / 精确恢复

Each selected checkpoint contains six model shards, six optimizer shards, six extra-state shards,
`data.pt`, tokenizer/config files and custom architecture source. Therefore optimizer-exact resume is a
**world-size-6 contract**. A model-only merge/export can later use another topology, but changing rank
count while claiming exact optimizer/RNG/data-cursor continuation is not allowed。

Reference policy:

- steps through global step 196 use each variant's frozen own step-98 export;
- steps 197-294 use each variant's frozen own step-196 export;
- the archive includes all four reference exports in addition to the 10 checkpoints;
- checkpoint optimizer moments and scheduler/data cursor must be loaded, not recreated;
- source scripts and runtime YAMLs are authoritative for scheduler rebasing and phase boundaries。

Before any continuation, run the repository checkpoint validator and compare the restored global step,
optimizer shards, `data.pt`, reference receipt, dataset hashes and trainable-tensor policy. A successful
model forward alone is not an exact-resume gate。

## Evaluation evidence / 评估材料

Included evidence covers corrected Math and OOD consolidated reports, selected parallel Math cells,
major-checkpoint OOD cells, HumanEval+ parser/protocol repairs, LiveCodeBench evidence,
GPQA-Diamond-Freeform generation plus the 126-question/1260-response manual audit, and all locally
synced compact receipts。Math cap-hit outputs remain historical protocol evidence; OOD corrected scores
and their correction reasons remain documented rather than silently replacing heritage files。

## Fresh-server acceptance checklist / 开跑前检查

1. Verify archive SHA-256, then all bundled source checksums.
2. Confirm six GPUs, driver, CUDA visibility and at least the recorded BF16 capability.
3. Install pinned core packages, patched veRL and this project in editable mode.
4. Verify original model SHA and decontaminated parquet/provenance hashes.
5. Run focused unit tests for registry/config, dataset order, checkpoint validation, vLLM custom routes,
   weight sync and resume gates.
6. Load one selected checkpoint on six ranks and confirm optimizer/data cursor/global step exactly.
7. Load the matching frozen reference export and verify its receipt.
8. Run one bounded rollout/update/restart canary before any long wave.
9. Keep the Mandatory PENDING registry visible; this archive does not waive unresolved architecture
   or Eval Parity Matrix obligations.

## Pending Obligations Carried Forward

The canonical registry remains
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`. At packaging time, unresolved
items remain carried forward exactly as registered; archive completeness is not evidence that Eval Parity
Matrix, pure-BF16 architecture paths, or registered SHS CausalLM obligations are complete。
