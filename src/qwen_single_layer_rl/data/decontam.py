from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


NORMALIZATION_VERSION = "nfkc_lower_latexspace_punct_ws_v2"
_LATEX_SPACE_RE = re.compile(r"\\(?:,|;|:|!|quad|qquad)\s*")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DecontamReport:
    input_count: int
    kept_count: int
    removed_count: int
    benchmark_hash_count: int
    exact_removed_count: int = 0
    near_removed_count: int = 0
    benchmark_problem_count: int = 0
    source_duplicate_removed_count: int = 0
    normalization_version: str = NORMALIZATION_VERSION


def normalize_problem(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = _LATEX_SPACE_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def hash_problem(text: str) -> str:
    return hashlib.sha256(normalize_problem(text).encode("utf-8")).hexdigest()


def _token_ngrams(text: str, n: int) -> frozenset[tuple[str, ...]]:
    return _normalized_token_ngrams(normalize_problem(text), n)


def _normalized_token_ngrams(normalized_text: str, n: int) -> frozenset[tuple[str, ...]]:
    tokens = normalized_text.split()
    if len(tokens) < n:
        return frozenset()
    return frozenset(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


class BenchmarkProblemIndex:
    def __init__(
        self,
        problems: Iterable[str],
        *,
        ngram_size: int = 8,
        jaccard_threshold: float = 0.50,
        containment_threshold: float = 0.80,
    ) -> None:
        if ngram_size <= 0:
            raise ValueError("ngram_size must be positive")
        self.ngram_size = ngram_size
        self.jaccard_threshold = jaccard_threshold
        self.containment_threshold = containment_threshold
        self.problems = tuple(str(problem) for problem in problems if str(problem).strip())
        self.hashes = frozenset(hash_problem(problem) for problem in self.problems)
        self._grams = tuple(_token_ngrams(problem, ngram_size) for problem in self.problems)
        inverted: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for problem_id, grams in enumerate(self._grams):
            for gram in grams:
                inverted[gram].append(problem_id)
        self._inverted = dict(inverted)

    def match_kind(self, problem: str) -> str | None:
        normalized = normalize_problem(problem)
        normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if normalized_hash in self.hashes:
            return "exact"
        grams = _normalized_token_ngrams(normalized, self.ngram_size)
        if not grams:
            return None
        candidates: set[int] = set()
        for gram in grams:
            candidates.update(self._inverted.get(gram, ()))
        for candidate_id in candidates:
            benchmark_grams = self._grams[candidate_id]
            overlap = len(grams & benchmark_grams)
            union = len(grams | benchmark_grams)
            shorter = min(len(grams), len(benchmark_grams))
            if overlap / union >= self.jaccard_threshold:
                return "near"
            if overlap / shorter >= self.containment_threshold:
                return "near"
        return None


def filter_decontaminated(
    records: Iterable[dict],
    benchmark_index: BenchmarkProblemIndex,
) -> tuple[list[dict], DecontamReport]:
    kept: list[dict] = []
    total = 0
    exact_removed = 0
    near_removed = 0
    duplicate_removed = 0
    seen_hashes: set[str] = set()
    for record in records:
        total += 1
        prompt = str(record.get("problem") or record.get("prompt") or record.get("question") or "")
        problem_hash = hash_problem(prompt)
        if problem_hash in seen_hashes:
            duplicate_removed += 1
            continue
        seen_hashes.add(problem_hash)
        match_kind = benchmark_index.match_kind(prompt)
        if match_kind == "exact":
            exact_removed += 1
        elif match_kind == "near":
            near_removed += 1
        else:
            kept.append(record)
    removed = exact_removed + near_removed + duplicate_removed
    return kept, DecontamReport(
        input_count=total,
        kept_count=len(kept),
        removed_count=removed,
        benchmark_hash_count=len(benchmark_index.hashes),
        exact_removed_count=exact_removed,
        near_removed_count=near_removed,
        benchmark_problem_count=len(benchmark_index.problems),
        source_duplicate_removed_count=duplicate_removed,
    )


def filter_exact_hashes(records: Iterable[dict], benchmark_hashes: set[str]) -> tuple[list[dict], DecontamReport]:
    kept: list[dict] = []
    total = 0
    removed = 0
    for record in records:
        total += 1
        prompt = str(record.get("problem") or record.get("prompt") or record.get("question") or "")
        if hash_problem(prompt) in benchmark_hashes:
            removed += 1
            continue
        kept.append(record)
    return kept, DecontamReport(
        input_count=total,
        kept_count=len(kept),
        removed_count=removed,
        benchmark_hash_count=len(benchmark_hashes),
        exact_removed_count=removed,
    )
