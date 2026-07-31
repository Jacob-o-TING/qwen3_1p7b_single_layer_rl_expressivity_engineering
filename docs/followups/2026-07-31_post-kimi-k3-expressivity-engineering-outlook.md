# Post-Kimi K3 Expressivity Engineering Outlook

**Date:** 2026-07-31
**Status:** Research agenda; not yet implemented or empirically validated

## Scope and provenance

This note records research directions formulated after reading the Kimi K3
technical report. It is intentionally separate from the repository's technical
report, which predates Kimi K3 and remains an unchanged record of the completed
Qwen3-1.7B experiments.

The earlier report already proposed per-token HyperGrid- or
HyperNetwork-style modulation, beginning with factorized forms rather than full
dynamic-weight generation. It did **not** specify the channel-wise diagonal
gate, low-rank full-matrix gate, or multi-LoRA HyperNetwork constructions below.
Those are new proposals in this follow-up.

The common design premise remains multiplication-first Expressivity
Engineering: fixed linear maps mix channels, whereas input-conditioned
multiplication, dynamic weights, and compositions of nonlinear components
create higher-order or context-dependent interactions.

## 1. Additive and multiplicative expert composition

### 1.1 Proposed family

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

### 1.2 Nearest literature and distinction

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

### 1.3 Load balancing after Kimi K3

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

## 2. SHS as a structured multi-LoRA HyperNetwork

The current SHS implementation uses HyperGrid-style structured modulation. A
lower-parameter successor can replace the grid with progressively richer
dynamic gates.

### 2.1 Channel-wise diagonal gate

For a projection `W` and activation `x`, generate one output-channel gate:

```text
g(x) in R^{d_out}
y = [1 + g(x)] elementwise_mul (W x).
```

This is a diagonal operator on the projection output. It is inexpensive,
precise at the channel level, and naturally exact-no-op when the HyperNetwork's
output head is zero-initialized. An input-channel analogue gates `x` before
`W`, and the two can be combined.

### 2.2 Low-rank full-matrix gate

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

## 3. Full-rank dynamic SwiGLU components

Another SHS successor can remove low-rank approximations and let a HyperNetwork
mix several full-rank SwiGLU components. Two materially different constructions
must not be conflated.

### 3.1 Weight-space mixture

For each SwiGLU projection `p` in `{gate, up, down}`:

```text
W_p,eff(x) = W_p,0 + sum_{e=1}^{E} alpha_e(x) W_p,e.
```

Input-dependent coefficients prevent this from collapsing into one static
matrix. With dense nonzero coefficients, every component can receive gradient
on every token, removing discrete-routing starvation. The price is dense
component compute or the cost of materializing a token-specific full matrix.

### 3.2 Output-space ensemble

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

### 3.3 Corrected complexity accounting

Let hidden width be `d`, FFN width be `d_ff = c d`, total experts be `E`, and
active experts per token be `k`.

| Construction | Per-token leading compute | Parameter scale | Main consequence |
| --- | ---: | ---: | --- |
| Dense FFN/SwiGLU | `Theta(d d_ff) = Theta(d^2)` | `Theta(d^2)` | One active component |
| Sparse top-`k` MoE | `Theta(k d d_ff) = Theta(k d^2)` | `Theta(E d^2)` | Sparse compute; routing and communication |
| Dense sum of `E` full experts | `Theta(E d^2)` | `Theta(E d^2)` | Every expert active; no sparse-compute saving |
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

## 4. Expressivity Engineering for attention replacement

Attention is already an EE mechanism: token-dependent weights modulate and mix
value representations. This suggests testing whether structured HyperNetworks
can replace a subset of attention layers rather than only modifying FFNs.

The relevant post-Kimi K3 baseline is
[Kimi Delta Attention](https://arxiv.org/abs/2510.26692), a recurrent
fast-weight mechanism with linear sequence-length scaling and a fixed-size
state instead of a sequence-length-growing KV cache. A useful successor must be
evaluated against KDA on more than asymptotic notation:

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
   KDA-style fast-weight state;
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

## 5. Proposed experimental order

1. Compare channel-wise diagonal, rank-one, rank-`R`, and HyperGrid SHS gates
   at matched parameter count and FLOPs.
2. Test residualized two-expert products against additive two-expert mixtures
   before scaling expert count.
3. Separate shared-expert and routed-expert fusion policies in a small MoE.
4. Compare dense all-expert training with QB-balanced sparse routing, measuring
   both load and semantic specialization.
5. Only after numerical and scaling gates pass, test full-rank dynamic SwiGLU
   bases and attention-layer replacements.

Every stage should begin as an exact functional no-op or a controlled
near-identity perturbation and should report parameter count, activated
parameters, FLOPs, communication, state memory, throughput, and matched-quality
metrics. These proposals are architectural hypotheses, not conclusions from
the completed Qwen3-1.7B study.

## Primary references

- Kimi Team. [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653), 2026.
- Kimi Team. [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692), 2025.
- Su et al. [CartesianMoE: Boosting Knowledge Sharing among Experts via Cartesian Product Routing in Mixture-of-Experts](https://aclanthology.org/2025.naacl-long.505/), NAACL 2025.
- Zhou et al. [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368), 2022.
- Schlag et al. [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174), 2021.
- Hinton. [Training Products of Experts by Minimizing Contrastive Divergence](https://doi.org/10.1162/089976602760128018), 2002.
