# 2026-07-11 SFT Compile Short Benchmark Record

## Scope

This record captures the formal single-GPU performance gate for the deterministic
Qwen3-1.7B layer-10 SFT pipeline. All four cases used the same model revision,
packed dataset cache, initialization seed, sample order, optimizer settings, and
20 timed optimizer steps after five warmup steps. Sequence length was 512 and
micro-batch size was one.

Hardware: one NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB).

Packed-cache contract:

- Source: NuminaMath-CoT 50k local JSONL.
- Packed sequences: 47,422.
- Overlong-prompt rows skipped deterministically: 93 of 50,000.
- Skipped-index hash: `ce52261d72bc555e92ef1f787de8bb0a4fc2c0697c0fb460b35ed3152d7a998f`.
- Cache manifest: `train_packed512_b686fa344aad5cd2.manifest.json`.

## Formal Results

| Case | Initial loss | Cold step | Median step | Assistant tok/s | Peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline eager | 0.197755 | 0.638 s | 0.079 s | 4,249.7 | 5.15 GB |
| Baseline compile | 0.197955 | 39.044 s | 0.024 s | 13,366.9 | 4.33 GB |
| SHS eager | 0.197755 | 0.808 s | 0.118 s | 2,809.2 | 7.37 GB |
| SHS compile | 0.197249 | 129.182 s | 0.091 s | 3,658.7 | 6.04 GB |

Derived comparisons:

- Baseline compile speedup over eager: 3.228x.
- SHS compile speedup over eager: 1.296x.
- SHS eager step-time overhead over baseline eager: 1.496x.
- SHS compiled step-time overhead over baseline compiled: 3.727x.
- Baseline compile cold-start break-even: approximately 717 optimizer steps.
- SHS compile cold-start break-even: approximately 4,800 optimizer steps.
- SHS eager exact-no-op initial-loss delta from baseline eager: exactly zero.
- SHS compile relative initial-loss delta from baseline eager: 0.256%.

The optimized SHS compile case captured one Dynamo graph with no graph-break
counter. Its separate deterministic Add/Multiply shuffle maps, architecture
dimensions, trainable parameter count, and exact eager no-op invariant were not
changed by the compile optimization.

## Decision

No case met the user-defined economically abnormal gate. The remote instance was
therefore not shut down. The 512-token result is a short performance gate rather
than a production runtime estimate: compile selection for the 2048-token run must
use a small 2048-token timing check plus the exact packed-sequence/optimizer-step
count.

Production observability remains sparse but whole-trajectory: JSONL and readable
metrics every optimizer step, with compact trainable checkpoints and validation
at approximately 10%, 25%, 50%, 75%, and 100%.

## Artifact Integrity

- Remote run root: `/root/autodl-tmp/qwen3_1p7b_single_layer_rl/runs/sft_compile_short_benchmark_20260711_formal_v1`.
- Remote log: `/root/autodl-tmp/qwen3_1p7b_single_layer_rl/logs/sft_compile_short_benchmark_20260711_formal_v1.log`.
- Local transfer archive SHA-256: `d5a34abee2713c5e4a8b9d72043088136ee3105cee6f7973465e9f1d7f20bdc7`.
- The 407 MiB disposable Inductor cache was excluded from the 19 KiB result archive.
