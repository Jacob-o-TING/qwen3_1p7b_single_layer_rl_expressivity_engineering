# 2026-07-12 OFT Read-Only Audit And Single-GPU Eval Projection

Date: 2026-07-12

Status: read-only structural and checkpoint audit complete; OFT evaluation is
still active, and the single-GPU vLLM evaluation estimate remains a projection
until a matched evaluator run is executed.

## OFT Placement And Freeze Contract

The active SFT configuration targets Layer 10 only. `QwenSwiGLUOFTWrapper`
wraps the original Layer-10 SwiGLU and applies three independent block-diagonal
Cayley rotations:

- gate projection input: 2,048 features, 32 blocks of 64;
- up projection input: 2,048 features, 32 blocks of 64;
- down projection input: 6,144 features, 96 blocks of 64.

The forward path is `base_linear(rotation(x))`. This is the input-centric
equivalent of transforming the pretrained weight by an orthogonal matrix, as
used by scalable OFT formulations. All rotations start at exact identity.

The original gate/up/down SwiGLU weights are frozen. Layer-10 attention
projections, Q/K norms, input RMSNorm, and post-attention RMSNorm remain
trainable. Layers other than 10, embeddings, and LM head remain frozen.

## Actual Final-Checkpoint Audit

The completed `step_00003916/trainable_state.pt` was inspected on CPU without
loading a model or using GPU memory:

- 11 tensors and 13,242,624 trainable parameters;
- three OFT tensors with 655,360 parameters;
- six attention tensors with 12,583,168 parameters;
- two RMSNorm tensors with 4,096 parameters;
- zero frozen base-MLP tensors;
- every value finite.

The learned rotations were modest and well conditioned:

| Projection | Raw RMS | Max abs | Mean rotation delta Frobenius | Max condition |
|---|---:|---:|---:|---:|
| gate | 0.002128 | 0.008872 | 0.5419 | 1.0113 |
| up | 0.002006 | 0.008804 | 0.5113 | 1.0087 |
| down | 0.001892 | 0.010661 | 0.4815 | 1.0110 |

Maximum orthogonality error was approximately `1.2e-6`. The observed benchmark
collapse is therefore not explained by a wrong target layer, accidentally
trainable base SwiGLU, non-finite checkpoint, non-orthogonal transform, or an
ill-conditioned Cayley solve.

OFT is a serious parameter-efficient adaptation family, but strong results in
other modalities and tasks do not guarantee this math-trajectory setting. The
current evidence is compatible with severe teacher-forced-loss versus
autoregressive-generation misalignment, an interaction between constrained MLP
updates and freely trained Layer-10 attention, or a train/eval reconstruction
defect not exposed by the static state audit. Required follow-up after the
active wave is:

1. complete OFT MATH-500/GSM8K/Olympiad/AMC@32 and greedy AMC;
2. compare initial, intermediate, and final checkpoints on a fixed greedy panel;
3. verify final-checkpoint trainer-path versus evaluator-path logits and tokens;
4. compare the OFT final checkpoint against the base model on response length,
   cap hits, extraction failure, and paired item outcomes;
5. do not classify the result as an inherent failure of canonical OFT until
   those gates pass.

## Current Serial Evaluation Cost

The active evaluator uses static Hugging Face microbatches of eight, not vLLM
continuous batching. Completed authoritative artifacts contain:

| Variant | Rows | Generated tokens | Mean tokens/row | Observed eval wall |
|---|---:|---:|---:|---:|
| Baseline | 3,774 | 2,471,003 | 654.7 | 4 h 45 m |
| SHS | 3,774 | 2,391,336 | 633.6 | 8 h 10 m |

The baseline wall interval was uninterrupted. The SHS interval is usable as an
end-to-end upper anchor but includes architecture overhead and may include more
orchestration variance.

## Single-GPU vLLM Projection

The matched long-decode benchmark measured 5,329.7 generated tokens/s for naive
Qwen vLLM and 2,262.8 tokens/s for SHS-reference vLLM at pressure 64 on one RTX
PRO 6000.

Applying those measured generation rates to the completed evaluation token
counts gives:

- baseline pure generation: approximately 7.7 minutes, versus 4 h 45 m serial;
- SHS pure generation: approximately 17.6 minutes, versus 8 h 10 m serial.

These are theoretical generation-only speedups of about 37x and 28x. Engine
load, prompt prefill, tokenizer work, JSONL persistence, extraction, grading,
and imperfect request pressure reduce end-to-end gains. A defensible planning
range on one GPU is:

| Variant path | Projected one-GPU vLLM eval | End-to-end speedup |
|---|---:|---:|
| Naive Qwen/baseline | 10-20 minutes | about 14-28x |
| SHS reference | 20-35 minutes | about 14-25x |

TriGLU and OFT require their own vLLM onboarding and matched long-response
measurements before assigning architecture-specific rates. Even at the lower
end of the observed acceleration range, replacing the current serial evaluator
should reduce a four-variant evaluation wave from roughly a day-scale process
to approximately one to three single-GPU hours. Four item-sharded TP=1 replicas
are a later post-production optimization and must preserve identical item,
sample, decode, extraction, and grading semantics.

## Dashboard Contract

The human-readable model summary now includes AMC greedy pass@1 directly for
every variant, including partial progress, alongside MATH-500, GSM8K,
OlympiadBench, and AMC Average@32. AMC greedy remains a decoding diagnostic and
is excluded from the four-benchmark math average.

## Geometry Preservation Does Not Guarantee Task Preservation

OFT restricts each adapted pretrained weight to an orthogonally transformed
subset that contains the identity. It preserves norms and pairwise angles in
the transformed weight-vector geometry. This is a meaningful protection
against unrestricted weight drift, but it does not imply monotonic downstream
accuracy or exact preservation of the network function after optimization.

Three distinctions matter in the present experiment:

1. The independent gate, up, and down rotations change the activation basis
   entering a nonlinear `SiLU(gate) * up` interaction. Geometry preservation of
   each projection does not preserve the composed SwiGLU function.
2. The current variant is not pure OFT-only adaptation. Layer-10 attention,
   Q/K norms, input RMSNorm, and post-attention RMSNorm receive unrestricted
   updates. Only the original SwiGLU weights are frozen and geometrically
   constrained through rotations.
3. Identity is a feasible point, but the SFT optimizer minimizes teacher-forced
   token loss. There is no guarantee that its selected feasible point improves
   external autoregressive mathematical correctness.

## Working Loss-Evaluation Conclusion

The present evidence strongly supports a model-selection mismatch:

- OFT final validation loss is `0.5893`, the lowest of the four variants;
- baseline validation loss is `0.6368`;
- OFT MATH-500 is `14.40%` versus baseline `57.60%`;
- the observed OFT GSM8K partial accuracy is approximately `28%` versus the
  completed baseline `78.32%`.

Subject to the still-required trainer/evaluator parity check, the working
conclusion is:

> Same-distribution teacher-forced SFT validation loss is not a reliable
> selection criterion for the desired autoregressive math-benchmark behavior.
> Lower loss may select a materially worse external checkpoint.

This does not isolate the dataset alone as the cause. The broader contract is
misaligned: Numina trajectory quality, full-solution imitation, teacher
forcing, same-distribution validation, autoregressive exposure bias, and the
chosen two-epoch schedule may all contribute. The result should therefore be
reported as an objective/data/validation-protocol mismatch rather than simply
"bad data" until the diagnostic controls separate them.

## Required Discriminating Controls

1. Build a checkpoint curve over initial, 10%, 25%, 50%, 75%, and final states
   with SFT validation loss, fixed-panel greedy accuracy, response length,
   cap-hit rate, and extraction failure. A falling validation loss paired with
   falling benchmark accuracy is direct evidence of proxy misalignment.
2. Run trainer-path versus evaluator-path logits and greedy-token parity on the
   final checkpoint before interpreting quality.
3. Run a pure OFT-only control that freezes Layer-10 attention and norms and
   trains only the three rotations. This distinguishes orthogonal FFN
   adaptation from its interaction with unrestricted Layer-10 updates.
4. Compare OFT against the untuned base and whole-layer baseline on identical
   item IDs with paired outcome categories and generation-length statistics.
5. Use external benchmark-aware checkpoint selection only as a declared
   diagnostic or future protocol; do not retroactively select the best final-
   test checkpoint from the current wave.
