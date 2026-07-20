# PENDING Mandatory Experiment Gates

Date created: 2026-07-12

Status: **PENDING REGISTRY - mandatory carry-forward**

This file is the canonical registry for important Qwen3-1.7B experiment work
that is not necessarily the immediate critical path but must not be forgotten.
Every later experiment plan must read this file and include a `Pending
Obligations Carried Forward` section listing every item whose status remains
`PENDING` or `IN_PROGRESS`.

An item may move to `COMPLETE` only when all of the following exist:

1. the approved implementation and experiment identity;
2. the required acceptance evidence and an authoritative dated record;
3. compact artifacts pulled to the local source-of-truth workspace;
4. source, tests, plans, and records committed and pushed;
5. an explicit completion update in this registry.

A partial implementation, source-only test, bounded smoke, or mention in
another plan does not remove an item from this registry. New plans may defer a
pending item, but must still name it, link this file, and state why it is not in
that plan's immediate execution scope.

## PENDING-01: Eval Parity Matrix

Status: **PENDING**

Purpose: establish decision-grade evaluator parity and a fast parallel
evaluation protocol before production comparisons rely on backend-specific
scores.

Required completion boundary:

- freeze the exact checkpoint, prompt, decoding, parser, scorer, seed, stop,
  cap, and sample-identity contracts for every matrix cell;
- finish the approved HF-versus-vLLM and initial-versus-trained checkpoint
  cells, using parallel/resumable execution rather than the retired serial
  evaluator;
- classify malformed generations separately from mathematically wrong answers;
- report exact sample-level agreement, extracted-answer agreement, score
  agreement, cap/stop behavior, missing rows, duplicates, and backend drift;
- choose and document either a strict-compatible evaluator or an all-model new
  vLLM protocol for later cross-architecture comparisons;
- preserve partial results, deterministic merge, and topology-neutral
  multi-GPU co-evaluation.

This item remains pending even if one historical HF cell or one vLLM shadow
cell has completed. The matrix and the owner-level protocol decision must both
be closed.

## PENDING-02: Pure-BF16 SHS And TriGLU Architecture Paths

Status: **PENDING**

Purpose: remove explicit FP32 custom-path execution from separately named SHS
and TriGLU runtime variants while retaining historical mixed-precision routes
as immutable controls.

The target is BF16, not IEEE FP16. Qwen3-1.7B is natively BF16 and the selected
vLLM serving path uses BF16 under `dtype=auto`.

Required completion boundary:

- implement explicit serializable `reference_fp32_custom` and
  `pure_bf16_custom` dtype policies without silently reinterpreting old
  checkpoints;
- keep custom parameters, forward inputs, intermediate activations, and output
  deltas in BF16 for the pure-BF16 route; report optimizer master state and
  accumulation dtype separately;
- preserve SHS and TriGLU exact initial no-op invariants;
- emit per-component dtype manifests, source-checkpoint hashes, and conversion
  receipts;
- pass HF reference drift, greedy, fixed-seed, forward/backward, memory, and
  throughput gates under preregistered BF16 tolerances;
- pass both architectures through their actual registered vLLM custom routes
  with semantic dispatch receipts and `fallback=false`;
- compare pure-BF16 throughput and memory against the historical reference
  policy and same-profile vanilla Qwen.

This item covers both SHS and TriGLU. Completing only one architecture does not
close it.

## PENDING-03: Registered SHS CausalLM Generation Route

Status: **PENDING**

Purpose: make the registered SHS vLLM wrapper satisfy the causal-LM generation
interface and prove that vLLM actually resolves it instead of the historical
generic `TransformersForCausalLM` route.

Required completion boundary:

- preserve the historical generic SHS route and measurements as an immutable
  control;
- implement the registered SHS causal-LM wrapper under a separately approved
  experiment identity;
- use the pinned vLLM registration/model-selection lifecycle and prove the
  resolved architecture/class;
- perform post-load device and dtype finalization at the correct lifecycle
  point;
- emit architecture, wrapper class, target layer, dimensions, seeds, dtype,
  backend, device, and `fallback=false` receipts;
- pass export/reload, exact no-op, HF reference, eight-token greedy,
  pressure-1/8/16, and matched long-decode gates;
- report registered-versus-generic differences as a new protocol result rather
  than rewriting historical SHS evidence.

The grouped SHS Triton backend is not part of this obligation and remains
unapproved after its strict full-logit gate failure.

## Carry-Forward Template

Every new experiment plan must contain a section equivalent to:

```text
## Pending Obligations Carried Forward

Canonical registry: docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md

- PENDING-01 Eval Parity Matrix: <in scope, deferred, or completed with record>
- PENDING-02 Pure-BF16 SHS And TriGLU: <in scope, deferred, or completed with record>
- PENDING-03 Registered SHS CausalLM Route: <in scope, deferred, or completed with record>
```

Do not copy stale status blindly. Read this registry at planning time and carry
forward only its current authoritative status.
