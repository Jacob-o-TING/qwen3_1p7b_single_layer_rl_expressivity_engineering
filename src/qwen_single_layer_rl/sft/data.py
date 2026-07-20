from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


IGNORE_INDEX = -100


class OverlongPromptError(ValueError):
    def __init__(self, source_index: int, prompt_tokens: int, max_length: int) -> None:
        super().__init__(
            f"SFT record {source_index} prompt has {prompt_tokens} tokens and cannot fit max_length={max_length}"
        )
        self.source_index = int(source_index)
        self.prompt_tokens = int(prompt_tokens)
        self.max_length = int(max_length)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas and pyarrow are required to read SFT parquet data") from exc
    rows = pd.read_parquet(path).to_dict(orient="records")
    for row in rows:
        messages_json = row.get("messages_json")
        if isinstance(messages_json, str) and messages_json:
            row["messages"] = json.loads(messages_json)
    return rows


def _normalized_messages(record: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    messages = record.get("messages")
    if isinstance(messages, list):
        normalized = [
            {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
            for item in messages
            if isinstance(item, dict)
        ]
        assistant_positions = [idx for idx, item in enumerate(normalized) if item["role"] == "assistant"]
        if assistant_positions:
            last = assistant_positions[-1]
            response = normalized[last]["content"].strip()
            prompt_messages = normalized[:last]
            if response and any(item["role"] == "user" and item["content"].strip() for item in prompt_messages):
                return prompt_messages, response

    problem = str(record.get("problem") or record.get("prompt") or record.get("question") or "").strip()
    solution = str(record.get("solution") or record.get("response") or "").strip()
    if not problem:
        raise ValueError("SFT record has no problem/user prompt")
    if not solution:
        raise ValueError(
            "SFT record has no full solution. Refusing to fall back to the RL final-answer field."
        )
    return [{"role": "user", "content": problem}], solution


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def _render_chat(tokenizer: Any, prompt_messages: list[dict[str, str]], response: str) -> tuple[list[int], list[int]]:
    full_messages = [*prompt_messages, {"role": "assistant", "content": response}]
    if getattr(tokenizer, "chat_template", None):
        prompt_ids = _token_ids(
            tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        full_ids = _token_ids(
            tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
    else:
        user_text = "\n".join(
            f"{item['role'].capitalize()}: {item['content']}" for item in prompt_messages
        )
        prompt_text = f"{user_text}\nAssistant:"
        full_text = f"{prompt_text} {response}"
        prompt_ids = _token_ids(tokenizer(prompt_text, add_special_tokens=True))
        full_ids = _token_ids(tokenizer(full_text, add_special_tokens=True))

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer chat template did not produce a stable prompt prefix for SFT masking")
    return prompt_ids, full_ids


@dataclass(frozen=True)
class EncodedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    source_index: int
    truncated_tokens: int


def encode_supervised_example(
    tokenizer: Any,
    record: dict[str, Any],
    source_index: int,
    max_length: int,
) -> EncodedExample:
    prompt_messages, response = _normalized_messages(record)
    prompt_ids, full_ids = _render_chat(tokenizer, prompt_messages, response)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and (not full_ids or full_ids[-1] != int(eos_id)):
        full_ids.append(int(eos_id))

    if len(prompt_ids) >= max_length:
        raise OverlongPromptError(source_index, len(prompt_ids), max_length)
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    truncated = max(0, len(full_ids) - max_length)
    if truncated:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError(f"SFT record {source_index} has no assistant tokens after truncation")
    return EncodedExample(tuple(full_ids), tuple(labels), int(source_index), truncated)


def pack_examples(
    examples: Iterable[EncodedExample],
    *,
    max_length: int,
    pad_token_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packed: list[dict[str, Any]] = []
    current_ids: list[int] = []
    current_labels: list[int] = []
    current_sources: list[int] = []
    truncated_tokens = 0
    example_count = 0

    def flush() -> None:
        if not current_ids:
            return
        real_length = len(current_ids)
        pad_count = max_length - real_length
        packed.append(
            {
                "input_ids": tuple(current_ids + [pad_token_id] * pad_count),
                "labels": tuple(current_labels + [IGNORE_INDEX] * pad_count),
                "attention_mask": tuple([1] * real_length + [0] * pad_count),
                "source_indices": tuple(current_sources),
                "non_padding_tokens": real_length,
                "assistant_tokens": sum(label != IGNORE_INDEX for label in current_labels),
            }
        )
        current_ids.clear()
        current_labels.clear()
        current_sources.clear()

    for example in examples:
        example_count += 1
        truncated_tokens += example.truncated_tokens
        if current_ids and len(current_ids) + len(example.input_ids) > max_length:
            flush()
        current_ids.extend(example.input_ids)
        current_labels.extend(example.labels)
        current_sources.append(example.source_index)
    flush()

    digest = hashlib.sha256()
    for item in packed:
        digest.update(json.dumps(item["source_indices"], separators=(",", ":")).encode("ascii"))
        digest.update(b"\n")
    return packed, {
        "example_count": example_count,
        "packed_sequence_count": len(packed),
        "max_length": max_length,
        "truncated_tokens": truncated_tokens,
        "packing_order_sha256": digest.hexdigest(),
    }


class PackedSFTDataset:
    def __init__(self, items: Sequence[dict[str, Any]]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def build_packed_dataset(
    tokenizer: Any,
    source_path: Path,
    *,
    max_length: int,
) -> tuple[PackedSFTDataset, dict[str, Any]]:
    records = read_records(source_path)
    encoded: list[EncodedExample] = []
    dropped_overlong_prompts: list[int] = []
    for index, record in enumerate(records):
        try:
            encoded.append(
                encode_supervised_example(tokenizer, record, source_index=index, max_length=max_length)
            )
        except OverlongPromptError:
            dropped_overlong_prompts.append(index)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
    items, manifest = pack_examples(encoded, max_length=max_length, pad_token_id=int(pad_token_id))
    manifest.update(
        {
            "source_path": str(source_path),
            "source_sha256": file_sha256(source_path),
            "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "unknown")),
            "label_policy": "assistant_only_ignore_index_-100",
            "packing_policy": "deterministic_sequential_greedy",
            "truncation_policy": "preserve_full_prompt_truncate_solution_skip_overlong_prompt",
            "dropped_overlong_prompt_count": len(dropped_overlong_prompts),
            "dropped_overlong_prompt_indices": dropped_overlong_prompts,
            "dropped_overlong_prompt_indices_sha256": hashlib.sha256(
                json.dumps(dropped_overlong_prompts, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
        }
    )
    return PackedSFTDataset(items), manifest


def packed_cache_path(
    cache_dir: Path,
    tokenizer: Any,
    source_path: Path,
    *,
    max_length: int,
) -> Path:
    signature = {
        "source_sha256": file_sha256(source_path),
        "tokenizer": str(getattr(tokenizer, "name_or_path", "unknown")),
        "max_length": int(max_length),
        "format_version": 1,
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{source_path.stem}_packed{max_length}_{digest}.pt"


def build_and_save_packed_cache(
    tokenizer: Any,
    source_path: Path,
    cache_path: Path,
    *,
    max_length: int,
) -> dict[str, Any]:
    import torch

    dataset, manifest = build_packed_dataset(tokenizer, source_path, max_length=max_length)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save({"items": dataset.items, "manifest": manifest}, temporary)
    temporary.replace(cache_path)
    cache_path.with_suffix(".manifest.json").write_text(
        json.dumps({**manifest, "cache_path": str(cache_path)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def load_packed_cache(cache_path: Path) -> tuple[PackedSFTDataset, dict[str, Any]]:
    import torch

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    return PackedSFTDataset(payload["items"]), dict(payload["manifest"])


def collate_packed_items(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "input_ids": torch.tensor([item["input_ids"] for item in items], dtype=torch.long),
        "labels": torch.tensor([item["labels"] for item in items], dtype=torch.long),
        "attention_mask": torch.tensor([item["attention_mask"] for item in items], dtype=torch.long),
        "non_padding_tokens": sum(int(item["non_padding_tokens"]) for item in items),
        "assistant_tokens": sum(int(item["assistant_tokens"]) for item in items),
        "source_indices": [list(item["source_indices"]) for item in items],
    }
