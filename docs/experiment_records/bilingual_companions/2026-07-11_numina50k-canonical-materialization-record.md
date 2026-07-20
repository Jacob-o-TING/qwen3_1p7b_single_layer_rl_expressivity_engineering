# 2026-07-11 NuminaMath 50K Canonical Materialization Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-11_numina50k-canonical-materialization-record.md](../2026-07-11_numina50k-canonical-materialization-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-11 NuminaMath 50K Canonical Materialization Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


## 范围 / Scope
This record pins the canonical SFT and RL training/validation materialization.
All architecture variants must consume these exact files with seed `20260707`,
the same shuffle setting, and the same dataloader seed.

## Benchmark Contract

- Qwen evaluation snapshot revision:
  `a45202bd16f1ec06f433442dc1152d0074773465`.
- MATH-500 snapshot revision:
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.
- Contract problem count: 2,534 unique problems across MATH-500, GSM8K,
  English text-only OlympiadBench, and AMC 2023.
- Contract SHA-256:
  `b6ccec96b4fe0f78de6bdf37e63fc15da73ec6590024a4b759963916709c913e`.
- Normalization: `nfkc_lower_latexspace_punct_ws_v2`.
- Near-match filter: token 8-grams, Jaccard threshold 0.50 or containment
  threshold 0.80.

## Source And Filtering

- Source: ModelScope mirror `AI-MO/NuminaMath-CoT`, train split, streaming.
- Source rows scanned: 859,494.
- Source exact duplicates removed before sampling: 65,648.
- Benchmark exact matches removed: 44.
- Benchmark near matches removed: 176.
- Eligible unique rows after filtering: 793,626.
- Final train rows: 50,000, selected with deterministic reservoir sampling.
- Final validation rows: 100, selected deterministically while excluding all
  train hashes, benchmark exact/near matches, and validation duplicates.

## Integrity Result

The independent materialized-data verifier passed all required checks:

- Train rows/unique hashes: 50,000 / 50,000.
- Validation rows/unique hashes: 100 / 100.
- Train benchmark exact/near matches: 0 / 0.
- Validation benchmark exact/near matches: 0 / 0.
- Train-validation exact overlap: 0.

Pinned SHA-256 values:

- Train JSONL:
  `723d9cacd4fed74efa871ef0d1ff59535f5882f23cf798ff09bb9b2b3a5dc4fc`.
- Train Parquet:
  `c515c67a3681bf3f466d87091b59264df76ec5f10e16d428ccebdf79a4f32b43`.
- Validation JSONL:
  `eba82e85f726e165ea95da8b1788ff4ee8944f864f34bedbbc3cd4d96e850392`.
- Validation Parquet:
  `f0fc2db5e2f1f2be6a1aa2faeb8e2bcc406613729ccaa633ab33374924090151`.
- Validation manifest:
  `903e9338942892b9384d686bda8f2af8132ab2454b64c590724994a7a56d942d`.
- Independent integrity report:
  `5345a5adf3c60d393e3d9cb0e007975c73ca7d13db8f648ef9948148db378c86`.

The canonical remote directory is
`data/numina_math_cot_50k_decontam_v3/`. Production configs and launchers must
not fall back to the older `data/numina_math_cot_50k/` materialization.

The RL-format copy is pinned under
`data/numina_math_cot_50k_decontam_v3_verl/`. It preserves canonical row order
before the seeded veRL sampler shuffle:

- Train output SHA-256:
  `16c145f165236a292140dd4fb86c4a0c4f7c6241a2390668cbf1d9364ded43d9`.
- Train problem-order SHA-256:
  `5f3c5c4b06bbd7b335b5536646dfcf31d4e70e7f31b3b3593f5e3d6c4383bbde`.
- Validation output SHA-256:
  `b6a3e2c0538d686258736fc8d0c655b296b52433205bbab2665f89347de6a3b3`.
- Validation problem-order SHA-256:
  `c1364fb447a4682f3d0d31220c7b3dedcb0d16043b835c97b2bae76f20159912`.

## 验证 / Verification
- Data-focused unit tests: 7 passed.
- Materialized audit: `passed: true`.
- The verifier now treats duplicates in either split as a hard failure rather
  than merely reporting them.

## Online Row Provenance

The complete lightweight selection ledger is committed under
`data_manifests/numina_math_cot_50k_decontam_v3/selected_rows.tsv`. Every one
of the 50,000 train rows and 100 validation rows records:

- materialized role and zero-based output index;
- online split and zero-based global source index;
- normalized problem SHA-256;
- normalized full-record SHA-256; and
- original Numina source category.

The upstream data payload is pinned to Hugging Face data commit
`e8b6aad745189763d0fd9521ac0844ce44675bef`, including the six Parquet LFS
object hashes in `configs/data/numina_math_cot_e8b6aad.json`. The actual fast
download route was the ModelScope mirror. Because that streaming API did not
expose an immutable mirror revision, the manifest additionally pins complete
observed problem-order and normalized-record-order SHA-256 digests for both
online splits.

Provenance verification result:

- Ledger rows: 50,100; unique `(role, materialized_index)` pairs: 50,100.
- Online train split: 859,494 rows, 50,001 selected references.
- Online test split: 100 rows, 99 selected references.
- Every selected full-record hash matched its materialized row.
- Train source problem-order SHA-256:
  `97fabdaa6cbf0ec1d37d1f508290399ab785acd59b50f500ffe30fff511024a5`.
- Train source normalized-record-order SHA-256:
  `43a0545ce4389e636dc3f7b1755238511cefe30e8264ca9ffc56602a5b5d659d`.
- Test source problem-order SHA-256:
  `57e348522d07289e1aaa104f2677d669ce4543d48374536c60f4b186ae85d0df`.
- Test source normalized-record-order SHA-256:
  `bb50ead2c881784eb1651a8e9edf0249100506b1fa4f9031df017d991d40893e`.
- Ledger SHA-256:
  `1097fdde429daf60eca6cbfb9b4e9f2f49ca5386133bd6618155087069a3968d`.

Validation is intentionally mixed-source: 99 rows came from the online test
split. One test candidate overlapped the canonical train sample, so the final
row came from the deterministic fallback candidate and maps to online
`train[157841]` (`cn_k12`). This is recorded as materialized
`validation[99]`, not incorrectly labeled as a test-split row.

### Normal SDK finding

ModelScope 1.33 recursively re-patched `HfFileSystem` when `MsDataset.load`
was called twice in one process. The provenance tool now performs one load
without a split, receives the train/test `IterableDatasetDict`, and streams
both members from that single load. This was an ordinary recoverable SDK issue,
not an economically abnormal condition.
