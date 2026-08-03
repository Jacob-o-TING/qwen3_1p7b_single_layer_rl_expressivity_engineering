# Expressivity Engineering Follow-up

**Date:** 2026-07-31
**Last updated:** 2026-08-03
**Status:** Retrospective interpretation and research agenda; the new
hypotheses are not yet independently validated

## Scope and provenance

This note records retrospective interpretation and research directions
formulated after reading the Kimi K3 technical report. It is intentionally
separate from the repository's technical report, which predates Kimi K3 and
remains an unchanged record of the completed Qwen3-1.7B experiments.

The earlier report already proposed per-token HyperGrid- or
HyperNetwork-style modulation, beginning with factorized forms rather than full
dynamic-weight generation. It did **not** specify the channel-wise diagonal
gate, low-rank full-matrix gate, or multi-LoRA HyperNetwork constructions below.
Those are new proposals in this follow-up.

The common design premise remains multiplication-first Expressivity
Engineering: fixed linear maps mix channels, whereas input-conditioned
multiplication, dynamic weights, and compositions of nonlinear components
create higher-order or context-dependent interactions.

## 1. Retrospective connection: the TriGLU bottleneck and LatentMoE

### 1.1 The architectural resemblance

The implemented TriGLU side network maps the Layer-10 hidden state through:

```text
x in R^2048
  -> z = A x in R^512
  -> three streams in R^2048
  -> element-wise three-stream product
  -> s in R^6144
  -> bounded multiplicative modulation of the base SwiGLU hidden state.
```

The `2048 -> 512` bottleneck was selected before the release of Kimi K3.
Earlier toy-model ablations had shown that adding a bottleneck could perform
better than a wider uncompressed side path, so it became part of the TriGLU
design on empirical rather than literature-derived grounds.

NVIDIA's
[LatentMoE](https://arxiv.org/abs/2601.18089) independently places a learned
down-projection and up-projection around the routed expert path:

```text
x in R^d -> z in R^ell -> routed expert computation in R^ell -> y in R^d.
```

The original motivation is hardware-software co-design: reducing routed width
lowers expert parameter traffic and all-to-all communication, and the saved
budget can be reinvested in more experts and a larger top-k. Kimi K3 later
adopts Stable LatentMoE, using a `7168 -> 3584` latent routed path at a much
larger scale.

The resemblance is substantive but not an identity. TriGLU compresses only an
always-active side modulator, retains the full-dimensional base SwiGLU path,
and performs no expert routing. LatentMoE compresses the routed MoE payload and
expert computation, while shared experts can remain in the original hidden
dimension.

### 1.2 A rank-constrained structural hypothesis

TriGLU's side return has the form:

```text
s(x) = f(Ax), where A maps R^2048 to R^512.
```

At differentiable points,

```text
J_s(x) = J_f(Ax) A,
rank[J_s(x)] <= 512.
```

The added modulator can therefore depend on the 2,048-dimensional input only
through 512 learned coordinates. This does not limit the rank of the complete
model, because the full base path remains present, but it does constrain the
new architectural degree of freedom. The side network must coordinate its
6,144-dimensional modulation through a shared low-dimensional latent
representation.

This supports several testable interpretations:

1. **Structured-feature pressure.** The bottleneck may force the side path to
   reuse shared latent factors instead of assigning independent freedom to
   every input direction.
2. **Nuisance suppression.** If useful activations lie near a lower-dimensional
   manifold, the learned projection may suppress directions that are weakly
   predictive or dominated by noise.
3. **Optimization regularization.** The factorization may reduce the effective
   search space and improve conditioning relative to an unconstrained side
   path.
4. **Capacity reinvestment.** Compute and parameters saved at the side input
   can support wider nonlinear streams after the bottleneck, paralleling
   LatentMoE's reinvestment of latent-space savings into expert diversity.

"Low-pass filter" is only an analogy unless a basis with a meaningful frequency
ordering is defined. Likewise, a bottleneck alone does not prove denoising.
The precise statement is rank-constrained learned compression; manifold
alignment, nuisance suppression, and improved reasoning are hypotheses.

The current evidence is suggestive rather than causal. Toy-model ablations
motivated the bottleneck, the Qwen3-1.7B TriGLU run produced matched-checkpoint
gains, and LatentMoE reports strong accuracy-efficiency results for a related
latent expert path. However, the Qwen experiment did not include a
parameter/FLOP-matched no-bottleneck TriGLU control. The present gains therefore
cannot yet be attributed specifically to the 512-dimensional compression.

### 1.3 Required ablation

A direct test should compare:

- the current `2048 -> 512 -> three 2048-dimensional streams -> 6144`
  side path;
- no compression before the three side streams;
- several bottleneck widths between 256 and 2,048;
- matched-parameter variants that trade bottleneck width against side-stream
  width; and
- a low-rank linear control without the three-stream product.

The evaluation should report quality, throughput, parameter count, local
Jacobian spectra, activation covariance rank, and sensitivity to input noise.
This would distinguish an optimization effect from a genuine
low-dimensional-structure effect.

## 2. Additive and multiplicative expert composition

### 2.1 Proposed family

A conventional sparse Mixture-of-Experts layer forms a router-weighted sum:

```text
y_add(x) = sum_{i in TopK(x)} p_i(x) E_i(x).
```

A multiplicative extension can instead combine selected expert
representations with a residualized element-wise product:

```text
m(x) = product_{j in M(x)} [1 + epsilon_j(x) tanh(E_j(x))]
y_mul(x) = x + beta(x) elementwise_mul [m(x) - 1].
```

The near-identity factors make the construction better conditioned than a raw
product of unbounded expert vectors. A mixed layer can allocate some experts
to the additive path and others to the multiplicative path:

```text
y(x) = x
     + sum_{i in A(x)} p_i(x) E_i(x)
     + beta(x) elementwise_mul
       {product_{j in M(x)} [1 + epsilon_j(x) tanh(E_j(x))] - 1}.
```

Shared and routed experts need not use the same rule. Candidate treatments
include:

1. shared experts as an additive anchor and routed experts as multiplicative
   refinements;
2. routed experts as an additive sparse path and shared experts as
   multiplicative gates;
3. separate additive and multiplicative shared-expert groups; or
4. multiplicative composition among several shared experts before they
   modulate the routed aggregate.

These variants should be compared under matched activated parameters, FLOPs,
communication volume, and router policy. Direct products require bounded
branches, normalization, and exact-no-op or near-identity initialization to
avoid exploding or vanishing activations and gradients. Kimi K3's SiTU-GLU was
itself introduced partly to bound the two factors of a multiplicative routed
FFN, reinforcing this numerical requirement.

### 2.2 Nearest literature and distinction

The closest direct precedent found in a targeted literature search is
[CartesianMoE](https://aclanthology.org/2025.naacl-long.505/). It derives a
Cartesian product of two sub-expert sets and realizes each composite expert by
**sequentially composing two routed MoE sub-layers**. The paper describes this
as knowledge sharing in a "multiplication" manner and reports improved routing
robustness. It is highly relevant, but it is not the same as taking an
element-wise product of multiple expert hidden representations within one
aggregation stage.

Classical
[Product of Experts](https://doi.org/10.1162/089976602760128018) models
multiply and renormalize probability distributions or energy factors. That is
also conceptually adjacent, but it is not a hidden-state MoE fusion rule.

The proposed additive/multiplicative split should therefore be presented as a
distinct hypothesis with known neighboring ideas, not as either an entirely
unprecedented use of expert multiplication or a reimplementation of
CartesianMoE.

### 2.3 Load balancing after Kimi K3

[Kimi K3](https://arxiv.org/abs/2607.24653) introduces Quantile Balancing (QB)
to choose expert-specific routing biases from score-margin quantiles, targeting
balanced loads without an auxiliary router loss. Its training system also uses
MoonEP with dynamic redundant experts to balance expert-parallel computation.
These mechanisms substantially weaken load imbalance as a reason to abandon
sparse routing.

They do not make the proposed dense or multiplicative expert families
redundant. QB targets token-count balance; it does not by itself guarantee
semantic diversity, equal learning progress, or freedom from expert
under-specialization. A dense all-expert path guarantees that every expert
receives a gradient when its coefficient is nonzero, but it sacrifices the
conditional-compute advantage of sparse MoE and can still learn redundant
experts.

## 3. SHS as a structured multi-LoRA HyperNetwork

The current SHS implementation uses HyperGrid-style structured modulation. A
lower-parameter successor can replace the grid with progressively richer
dynamic gates.

### 3.1 Channel-wise diagonal gate

For a projection `W` and activation `x`, generate one output-channel gate:

```text
g(x) in R^{d_out}
y = [1 + g(x)] elementwise_mul (W x).
```

This is a diagonal operator on the projection output. It is inexpensive,
precise at the channel level, and naturally exact-no-op when the HyperNetwork's
output head is zero-initialized. An input-channel analogue gates `x` before
`W`, and the two can be combined.

### 3.2 Low-rank full-matrix gate

A more expressive gate modulates individual entries of `W` while generating
only low-rank factors:

```text
G(x) = sum_{r=1}^{R} a_r(x) b_r(x)^T
W_eff(x) = W elementwise_mul [1 + G(x)].
```

`R=1` gives a rank-one dynamic gate; larger `R` gives a multi-rank gate. A
related additive multi-LoRA HyperNetwork is:

```text
DeltaW(x) = sum_{l=1}^{L} alpha_l(x) A_l B_l
W_eff(x) = W + DeltaW(x).
```

The multiplicative and additive forms can coexist. The important comparison is
not merely "HyperGrid versus LoRA," but granularity and rank under matched
parameter and FLOP budgets:

- channel-wise diagonal gating;
- rank-one full-matrix gating;
- rank-`R` full-matrix gating;
- a mixture of several fixed LoRA bases with dynamic coefficients; and
- the existing shuffled HyperGrid.

A literal full dynamic `d_out x d_in` gate is not parameter-efficient. The
parameter reduction comes from diagonal structure, low-rank factorization,
basis sharing, or a combination of them.

## 4. Full-rank dynamic SwiGLU components

Another SHS successor can remove low-rank approximations and let a HyperNetwork
mix several full-rank SwiGLU components. Two materially different constructions
must not be conflated.

### 4.1 Weight-space mixture

For each SwiGLU projection `p` in `{gate, up, down}`:

```text
W_p,eff(x) = W_p,0 + sum_{e=1}^{E} alpha_e(x) W_p,e.
```

Input-dependent coefficients prevent this from collapsing into one static
matrix. With dense nonzero coefficients, every component can receive gradient
on every token, removing discrete-routing starvation. The price is dense
component compute or the cost of materializing a token-specific full matrix.

### 4.2 Output-space ensemble

Alternatively, retain separate full SwiGLU experts:

```text
E_e(x) = W_down,e [
    SiLU(W_gate,e x) elementwise_mul (W_up,e x)
]
y(x) = sum_e alpha_e(x) E_e(x).
```

Because each component contains its own nonlinearity, this does not collapse
into one linear map. With dense coefficients it is essentially a soft, dense
MoE augmented by SHS-style dynamic weighting or multiplicative cross-expert
interactions.

Simply summing static linear matrices **before** all nonlinearities is not an
expressivity gain:

```text
sum_e W_e x = (sum_e W_e) x.
```

That form is exactly one linear projection and can be cached as such.

### 4.3 Corrected complexity accounting

Let hidden width be `d`, FFN width be `d_ff = c d`, total experts be `E`, and
active experts per token be `k`.

For the common illustrative choice `d_ff = 4d`, one projection from `d` to
`4d` has a `4d x d` weight matrix. Applying it to **one token vector** requires:

```text
number of output coordinates x dot-product length
= 4d x d
= 4d^2 multiply-accumulates,
```

not `4d^3`. The two SwiGLU input projections and one output projection require:

```text
gate: 4d^2
up:   4d^2
down: 4d^2
total: 12d^2 multiply-accumulates per token and per expert.
```

If one multiply-accumulate is counted as two FLOPs, the same operation is
approximately `24d^2` FLOPs. For `n` tokens and `k` active experts, the leading
expert compute is `12 n k d^2` multiply-accumulates. A cubic expression arises
only after an additional assumption such as `n = Theta(d)` or
`k = Theta(d)`.

| Construction | Per-token leading compute | Parameter scale | Main consequence |
| --- | ---: | ---: | --- |
| Dense FFN/SwiGLU | about `12d^2` MACs when `d_ff=4d` | `Theta(d^2)` | One active component |
| Sparse top-`k` MoE | about `12k d^2` MACs | `Theta(E d^2)` | Sparse compute; routing and communication |
| Dense sum of `E` full experts | about `12E d^2` MACs | `Theta(E d^2)` | Every expert active; no sparse-compute saving |
| Element-wise product of `m` expert outputs | `Theta(m d^2)` plus `Theta(m d)` fusion | `Theta(m d^2)` | Higher-order fusion; conditioning risk |
| Static pre-sum of `E` linear matrices | one-time `Theta(E d^2)`, then `Theta(d^2)` | reducible to `Theta(d^2)` | Collapses to one matrix |
| Input-conditioned mixture of `E` full matrices | generally `Theta(E d^2)` | `Theta(E d^2)` | Dynamic full-rank basis; expensive unless structured |

Standard sparse MoE is therefore **not intrinsically cubic in model width**.
Its token compute is quadratic for fixed `k` and `d_ff/d`. A `Theta(d^3)` term
appears only under an additional scaling assumption such as `k = Theta(d)` or
`E = Theta(d)` with all experts active, or when an actual dense `d x d`
matrix-matrix multiplication is performed.

The corrected claim is that structured SHS variants may avoid the
`Theta(E d^2)` parameter, compute, communication, or memory cost associated
with many full experts. They do not generically reduce a standard fixed-top-`k`
MoE from cubic to quadratic compute.

### 4.4 Why pre-adding full matrices does not replace sparse MoE

Suppose a HyperNetwork produces one set of coefficients and forms one effective
SwiGLU by adding `E` full matrix triplets. For one coefficient set, the rough
work is:

```text
form three effective matrices: about 12 E d^2 element-wise scale/add operations
apply the effective SwiGLU:    about 12 n d^2 multiply-accumulates for n tokens
```

The corresponding expression is therefore proportional to:

```text
12 E d^2 + 12 n d^2,
```

not `12 E d^2 + 12 d^3`. This can be cheaper than applying `k` complete experts
to all `n` tokens when the **same** effective matrices are reused across enough
tokens:

```text
E + n < k n.
```

That is a real amortization regime for sequence-level, segment-level, or
batch-level HyperNetwork coefficients. It is not a sparse-MoE replacement:
sharing one coefficient set across many tokens gives up the token-wise
conditional computation that defines MoE.

There are three important boundaries:

1. If the coefficients are static, the matrices can be merged once, but the
   construction collapses to one ordinary SwiGLU and adds no dynamic
   expressivity.
2. If every token has different coefficients, matrix aggregation must be
   repeated for every token. The cost becomes approximately
   `12 n E d^2 + 12 n d^2`, eliminating the proposed amortization and creating
   substantial memory traffic.
3. Pre-adding matrices before the nonlinearity is not functionally equivalent
   to summing `E` independently evaluated SwiGLU experts. The former produces
   one context-conditioned SwiGLU; the latter retains `E` distinct nonlinear
   expert functions.

The original replacement hypothesis is therefore rejected. In the intended
per-token dynamic setting, `E` full matrix bases cost approximately
`12 n E d^2`, while a sparse top-`k` MoE costs approximately
`12 n k d^2`; modern MoEs deliberately choose `E` much larger than `k`.
Pre-adding all expert matrices consequently defeats MoE's defining
compute-saving purpose.

A context-shared full-rank HyperNetwork basis mixture may still be useful as a
separate EE module or controlled dense baseline. It can generate effective
weights infrequently enough to amortize matrix formation, and SHS-style
multiplication can restore interactions that a static matrix sum would lose.
It should be described as complementary to sparse MoE, not as its replacement.

## 5. Expressivity Engineering for attention replacement

Attention is already an EE mechanism: token-dependent weights modulate and mix
value representations. This suggests testing whether structured HyperNetworks,
selective state-space models, or hybrids can replace a subset of attention
layers rather than only modifying FFNs.

Two important non-softmax baselines are
[Kimi Delta Attention](https://arxiv.org/abs/2510.26692) (KDA) and
[Mamba](https://arxiv.org/abs/2312.00752). KDA is a recurrent fast-weight
mechanism with linear sequence-length scaling and a fixed-size state instead
of a sequence-length-growing KV cache. Mamba is a selective state-space model
whose recurrence can be written schematically as

\[
h_t = \bar A_t h_{t-1} + \bar B_t x_t,
\qquad
y_t = C_t h_t.
\]

In Mamba, the discretization step `Delta_t` and the write/read terms `B_t` and
`C_t` depend on the current input. Consequently, the effective transition,
write, and read operators are token-conditioned even though Mamba does not
regenerate every parameter, such as the underlying continuous-time `A`, at
every step. Under the EE lens, this selective recurrence is another instance
of input-dependent multiplicative computation: the token helps determine how
its information is retained, forgotten, written into state, and read out.
Mamba therefore provides both an attention alternative and an established
example of dynamic operators with linear sequence scaling and fixed-size
recurrent state during autoregressive inference.

A useful successor must be evaluated against KDA and Mamba on more than
asymptotic notation:

- quality at matched training FLOPs and activated parameters;
- recurrent-state size and memory bandwidth;
- prefill and decode throughput with production kernels;
- numerical stability and parallel training efficiency;
- long-context retrieval and overwrite behavior; and
- compatibility with hybrid full-attention layers.

A stateless per-token HyperNetwork can generate a transient low-rank or
structured transformation without a persistent recurrent state, but it also
has no mechanism for preserving long-range information unless it consumes a
context summary or state. Once such a state is introduced, the comparison is
again between alternative fast-weight memories.

Promising controlled variants include:

1. a HyperNetwork that generates low-rank write and erase updates to a
   KDA- or Mamba-style recurrent state;
2. channel-wise or rank-`R` dynamic gates over the recurrent transition;
3. nonlinear multi-branch write rules whose products are residualized and
   bounded;
4. cross-layer sharing of generated low-rank bases; and
5. hybrid stacks in which the proposed module replaces only selected attention
   layers.

The phrase "no permanent memory cost" should be avoided. A generated
transformation may be transient during inference, but its generation still
costs compute and activation bandwidth; training must retain intermediates for
backpropagation; and any long-range recurrent mechanism requires persistent
state. The defensible target is **no sequence-length-growing KV cache**, not
zero memory.

## 6. Frequency-shaped EE as an alternative to recursive depth

### 6.1 What HRM-Text actually repeats

[HRM](https://arxiv.org/abs/2506.21734) and
[HRM-Text](https://arxiv.org/abs/2605.20613) motivate hierarchical recurrence
from multi-timescale processing in the brain: a fast `L` module performs local
iterative refinement, while a slower `H` module maintains a more stable,
high-level context. This is a multi-timescale functional analogy, not evidence
that the architecture literally reproduces measured neural oscillations.

The exact HRM-Text schedule is important. It is not `(L, L, H) x 2`.
HRM-Text uses two high-level cycles, each containing three `L` updates followed
by one `H` update:

```text
(L, L, L, H) x 2
```

The paper denotes this configuration `H2L3`. One forward pass therefore reuses
the `L` module six times and the `H` module twice. Because each module contains
approximately half of the non-embedding recurrent-core parameters, the paper
counts the eight module applications as four recursion-equivalent units of
compute. The EE schedule below consequently adopts the verified `3:1` cadence
rather than retaining the earlier `2:1` simplification.

Recurrence exposes a fundamental compute-capacity tradeoff. Reapplying a block
increases FLOPs, activation traffic, and latency without creating new learned
parameters. Parameter sharing may improve algorithmic reuse, iterative
refinement, generalization, or reasoning per parameter, but it does not add
independent weight storage. In a controlled setting that removes
generalization, Morris et al. estimate the memorization capacity of GPT-style
models at approximately 3.6 bits per parameter. This is an empirical scaling
result for their setup, not a universal constant for all architectures or a
proof that benchmark knowledge is stored at a fixed rate. Nevertheless, it
sharpens the concern: compared with an untied model using the same active
forward compute, a tied recurrent model has fewer independent parameters in
which to encode factual information. Recurrence may therefore exchange some
potential knowledge-storage capacity for more computation over the knowledge
and algorithms already represented.

### 6.2 An operational EE level and a depth-domain frequency

The technical report defines EE through a multiplication-first primitive: a
fixed linear map performs channel mixing, whereas an input-derived quantity
multiplies the same representation, another transform of it, or dynamically
generated weights. It does not define an integer-valued EE level. This
follow-up therefore introduces the following provisional operational
definition:

- an ordinary SwiGLU block is `Level = 1`;
- each additional **sequential** input-conditioned multiplicative composition
  along a computational path raises the level by one; and
- parallel modulation sites are recorded separately rather than automatically
  added to the sequential level.

The sequential qualifier matters. SHS can use one HyperNetwork output to
modulate the `gate`, `up`, and `down` SwiGLU projections. It is unclear whether
this should count as one higher-level control event or three EE increments.
Two measurements should therefore be reported: multiplicative path depth
`D_mul`, the maximum number of sequential input-conditioned multiplications on
a path, and modulation-site count `S_mod`, the number of distinct tensors or
projections modulated. SHS may have one controller with `S_mod = 3`, while its
effective `D_mul` depends on the exact factorization. Whether one-site and
three-site modulation behave like different "levels" is an empirical question
for matched parameter/FLOP ablations, not something notation can settle.

With this convention, an EE level schedule across model depth can be treated
as a generalized frequency envelope. "Frequency" here means structured
variation along the depth axis, not Fourier frequency or literal neural
oscillation. A low-amplitude, short-period block can be defined as:

```text
L_EE = [1, 2, 2, 1]
```

A broader, higher-amplitude block can be defined as:

```text
H_EE = [1, 2, 3, 4, 3, 2, 1]
```

The revised proposal instantiates one cycle as three low-amplitude blocks
followed by one high-amplitude block:

```text
L_EE, L_EE, L_EE, H_EE
= [1, 2, 2, 1,
   1, 2, 2, 1,
   1, 2, 2, 1,
   1, 2, 3, 4, 3, 2, 1]
```

and executes that cycle twice. This preserves the intended analogy: repeated
lower-amplitude detail processing is periodically coordinated by a broader,
higher-level abstraction block. In compact notation, the complete schedule is:

```text
(L_EE, L_EE, L_EE, H_EE) x 2.
```

### 6.3 A middle-loop implementation and its compute boundary

The recurrent region should initially be restricted to the middle of the
stack. This is supported by two separate observations. *Is One Layer Enough?*
finds that high-contribution RL layers concentrate in the middle; for
Qwen3-1.7B, Layer 10 of 28 is the strongest reported single-layer math result
and lies roughly 36--39% into the stack depending on the indexing convention.
[Training-Free Looped Transformers](https://arxiv.org/abs/2605.23872) finds
that the centers of the best loop windows lie around 45--60% depth for seven
of nine tested checkpoints. Its Qwen3-1.7B-Base configuration loops Layers
12--15 with `K = 2` and a damped-Euler treatment. Together these findings
motivate a provisional 35--60% search envelope, not a universal biological or
architectural law.

The training-free method also supplies an important stability warning. Naive
reapplication often degrades performance because a pretrained residual block
is not necessarily contractive. The paper interprets a pre-norm block as a
forward-Euler step and replaces one large residual update with smaller damped
substeps; for Qwen3-1.7B, the reported reference uses `K = 2` and
`alpha = 0.5`. Native training may learn a recurrent block directly, but it
should still compare undamped, damped, normalized, and residual-scaled forms.

If a middle window accounts for a fraction `w` of otherwise equal-cost layers
and is executed `K` total times, its first-order compute multiplier is:

```text
C_loop / C_base = 1 + (K - 1) w.
```

Thus a 25% middle window executed twice gives approximately `1.25x` total
forward compute. This estimate is valid only when repeated layers have the
same cost as their baseline counterparts and ignores orchestration overhead.
The proposed higher EE levels themselves add branches or multiplicative
operators, so their true cost must instead use the measured cost `c(level)`:

```text
C_EE / C_base
= [C_outside + sum_j c(level_j)] / [N c(1)].
```

The `1.25x` figure is therefore the recurrence-only anchor, not a guaranteed
end-to-end cost for the frequency-shaped EE architecture.

For an initial FFN-focused experiment, attention-mode variation can be held
approximately constant by assigning Full and Linear Attention at head level.
The architecture matching that description is
[HydraHead](https://arxiv.org/abs/2606.20097), which hybridizes Full and Linear
Attention along the head axis; it is distinct from the 2022
[Hydra Attention](https://arxiv.org/abs/2209.07484) mechanism for vision
transformers. HydraHead makes the full/linear head fraction a design axis, but
does not establish that an arbitrary ratio can be changed without retraining
or distillation. Assuming a trained head-level hybrid allows the desired ratio
to be represented in every layer, the first EE recurrence study can hold that
ratio fixed and isolate the FFN level envelope.

### 6.4 Unrolling the loop into independent parameters

Once a frequency envelope has been defined without requiring literal module
identity, recurrence can be removed entirely. Instead of applying the same
`L_EE` and `H_EE` parameters across two cycles, instantiate two untied copies
with the same level schedule. The forward graph then preserves the proposed
multi-timescale pattern but becomes an ordinary deeper network:

```text
cycle 1: unique L_EE(1), L_EE(2), L_EE(3), H_EE(1)
cycle 2: unique L_EE(4), L_EE(5), L_EE(6), H_EE(2)
```

This gives six unique `L_EE` blocks and two unique `H_EE` blocks. At equal
operator count, tied recurrence and untied unrolling have approximately the
same forward FLOPs, while the untied model has more independent parameters.
This maximizes independent weight capacity per unit of active compute relative
to the tied version and may preserve more factual knowledge if knowledge
storage is parameter-limited.

The advantage is not free. Untying increases checkpoint size, optimizer state,
training memory, communication, and data required to train each parameter. It
also discards recurrence-specific inductive biases: iterative refinement under
one shared algorithm, depth extrapolation, and adaptive computation by running
more cycles. More parameters do not guarantee more usable knowledge, and an
untied model can undertrain or overfit. The proposal is therefore a controlled
alternative to recurrence, not a claim that unrolling dominates HRM.

The primary target is native pretraining, where the level envelope and
normalization can co-adapt from initialization. A second path can retrofit an
existing model by cloning middle blocks into untied copies, initializing each
copy from its source layer, adding exact-no-op EE branches, and briefly
continued-training with symmetry-breaking regularization. Both paths are worth
testing, but their conclusions must remain separate.

### 6.5 Attention patterns as a second generalized frequency

The same depth-domain perspective can include attention. Interleaving Full
Attention with Sparse or Linear Attention creates another structured pattern:
Full Attention supplies unconstrained token-pair interactions and can play an
`H`-like role, while efficient attention supplies cheaper recurrent or sparse
mixing and can play an `L`-like role. This is an architectural analogy, not a
claim that the modules have the same function as HRM's `H` and `L` states.

Kimi K3 provides a concrete layer-wise example. Its 93 attention layers contain
69 KDA layers and 24 Gated MLA layers. Full Attention therefore occupies
`24 / 93 = 25.8%` of the stack, approximately one full-attention layer per four
total attention layers. The strict pairwise ratio is
`Full:KDA = 24:69`, approximately `1:2.875`, conventionally summarized as a
three-KDA-to-one-full pattern rather than `Full:KDA = 1:4`.

Because both paths are attention mechanisms and KDA is itself an
input-conditioned EE mechanism, their EE difference cannot be reduced to a
single scalar level without measurement. Nevertheless, a hybrid that varies
only attention type while leaving FFNs unchanged expresses frequency on the
token-mixing axis but little or none on the FFN expressivity axis. Under the
present hypothesis, that gives a lower-amplitude depth signal than jointly
varying the multiplicative FFN level. This is plausible but unproven; Full
Attention can have a large functional advantage on exact retrieval even when
its algebraic "level" appears similar.

HydraHead changes this picture by mixing Full and Linear Attention inside each
layer. If its head ratio is held constant over depth, the attention frequency
is encoded as within-layer composition rather than alternating layer types,
leaving the FFN `Level` schedule as the main depth-domain wave. A broader study
should vary both axes jointly:

```text
depth position -> [FFN EE level, full-attention head fraction].
```

This would test whether high-amplitude EE blocks should coincide with more
full-attention capacity, alternate with it, or remain statistically
independent.

### 6.6 Required comparisons

The minimum native-training comparison should include:

1. a standard non-recurrent Transformer with constant `Level = 1`;
2. tied `(L_EE, L_EE, L_EE, H_EE) x 2` recurrence;
3. an untied unrolling of that schedule at matched active FLOPs;
4. a middle-window `K = 2` damped-Euler control without added EE levels;
5. flat-depth EE controls matched for parameters and FLOPs; and
6. fixed versus depth-varying Full/Linear Attention allocations.

Every comparison should report independent and activated parameters, training
and inference FLOPs, wall-clock throughput, optimizer and checkpoint memory,
factual-knowledge benchmarks, controlled memorization capacity, reasoning,
generalization, and sensitivity to additional recurrent steps. This separates
three questions that aggregate accuracy would blur: whether recurrence uses a
fixed parameter set more effectively, whether EE level schedules improve
reasoning at matched compute, and whether untied unrolling recovers knowledge
capacity without losing the multi-timescale inductive bias.

## 7. Proposed experimental order

1. Compare channel-wise diagonal, rank-one, rank-`R`, and HyperGrid SHS gates
   at matched parameter count and FLOPs.
2. Test residualized two-expert products against additive two-expert mixtures
   before scaling expert count.
3. Separate shared-expert and routed-expert fusion policies in a small MoE.
4. Compare dense all-expert training with QB-balanced sparse routing, measuring
   both load and semantic specialization.
5. Compare tied middle recurrence, damped-Euler recurrence, and untied
   frequency-shaped EE schedules at matched active FLOPs.
6. Only after numerical and scaling gates pass, test full-rank dynamic SwiGLU
   bases and attention-layer replacements.

Every stage should begin as an exact functional no-op or a controlled
near-identity perturbation and should report parameter count, activated
parameters, FLOPs, communication, state memory, throughput, and matched-quality
metrics. These proposals are architectural hypotheses, not conclusions from
the completed Qwen3-1.7B study.

## Primary references

- Kimi Team. [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653), 2026.
- Elango et al. [LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts](https://arxiv.org/abs/2601.18089), 2026.
- Kimi Team. [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692), 2025.
- Gu and Dao. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752), 2023.
- Wang et al. [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734), 2025.
- Wang et al. [HRM-Text: Efficient Pretraining Beyond Scaling](https://arxiv.org/abs/2605.20613), 2026.
- Morris et al. [How Much Do Language Models Memorize?](https://arxiv.org/abs/2505.24832), 2025.
- Chen et al. [Training-Free Looped Transformers](https://arxiv.org/abs/2605.23872), 2026.
- Zhang et al. [Is One Layer Enough? Training a Single Transformer Layer Can Match Full-Parameter RL Training](https://arxiv.org/abs/2607.01232), 2026.
- Tan et al. [HydraHead: From Head-Level Functional Heterogeneity to Specialized Attention Hybridization](https://arxiv.org/abs/2606.20097), 2026.
- Bolya et al. [Hydra Attention: Efficient Attention with Many Heads](https://arxiv.org/abs/2209.07484), 2022.
- Su et al. [CartesianMoE: Boosting Knowledge Sharing among Experts via Cartesian Product Routing in Mixture-of-Experts](https://aclanthology.org/2025.naacl-long.505/), NAACL 2025.
- Zhou et al. [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368), 2022.
- Schlag et al. [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174), 2021.
- Hinton. [Training Products of Experts by Minimizing Contrastive Divergence](https://doi.org/10.1162/089976602760128018), 2002.
