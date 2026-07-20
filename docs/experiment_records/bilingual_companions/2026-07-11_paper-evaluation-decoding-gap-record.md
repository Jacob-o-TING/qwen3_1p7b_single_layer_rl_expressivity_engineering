# 2026-07-11 Paper Evaluation Decoding Gap Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-11_paper-evaluation-decoding-gap-record.md](../2026-07-11_paper-evaluation-decoding-gap-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-11 Paper Evaluation Decoding Gap Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


## 发现 / Finding
The decoding protocol used for the paper's reported Qwen3 evaluation scores is
underdetermined from arXiv v2 of *Is One Layer Enough? Training A Single
Transformer Layer Can Match Full-Parameter RL Training* (arXiv:2607.01232).

Section 3.1 states that the primary Qwen3 math evaluation uses MATH500, GSM8K,
OlympiadBench, and AMC. Appendix B states only that AMC is small and therefore
uses Average@32. The paper does not specify, for these evaluations:

- greedy versus sampled generation;
- temperature;
- top-p or top-k;
- number of generations for MATH500, GSM8K, or OlympiadBench;
- prompt template and chat-template details;
- stop conditions or finish-reason handling;
- the evaluation harness or executable configuration.

Searching the arXiv HTML source finds no occurrences of `greedy` or
`temperature`. `Average@32` appears only in the AMC benchmark-description
sentence. The Appendix A hyperparameter tables describe RL training and rollout
settings such as maximum response length; they do not resolve final-evaluation
decoding.

Paper source:

```text
https://arxiv.org/html/2607.01232
```

No official evaluation repository or configuration has been identified as of
this record. This is an absence-of-evidence statement, not a claim that no such
artifact can later be released.

## Consequences For This Reproduction

The current local evaluator uses:

```text
MATH500, GSM8K, OlympiadBench: do_sample=false (greedy)
AMC Average@32: temperature=1.0, top_p=1.0, seeded multinomial sampling
```

Those settings are explicit, reproducible project choices made under missing
paper details. They must be described as `paper-aligned`, not
`paper-identical`, and must not be attributed to the paper.

In particular, the contrast between our stronger greedy results on MATH500,
GSM8K, and OlympiadBench and our weak sampled AMC result is valid evidence about
our checkpoint under our harness. It is not evidence that the paper obtained
its harder-benchmark scores using greedy decoding.

## Required Controls

Mechanistic conclusions should rely on decoding-matched local controls:

1. SHS sampled AMC versus SHS greedy AMC.
2. Whole-layer baseline sampled AMC versus whole-layer baseline greedy AMC.
3. Preferably, untuned Qwen3-1.7B-Base under both identical local protocols.

Comparisons to the paper's numerical rows remain orientation only until the
authors publish exact evaluation settings or a reproducible official evaluator.
