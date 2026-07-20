from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationSpec:
    max_new_tokens: int
    temperature: float
    do_sample: bool
    top_p: float
    seed: int | None
    stop: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedText:
    text: str
    generated_tokens: int | None = None
    finish_reason: str | None = None


class GeneratorBackend(Protocol):
    actual_backend: str
    engine_load_seconds: float

    def generate(
        self,
        conversations: list[list[dict[str, str]]],
        spec: GenerationSpec,
        identities: list[dict[str, Any] | None],
    ) -> list[GeneratedText]: ...


class JsonlReceiptWriter:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._completed = 0
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self._lock:
            payload = {"timestamp_unix": time.time(), **event}
            if event.get("event") == "generation_completed":
                self._completed += 1
                payload["completed_count"] = self._completed
            line = json.dumps(payload, sort_keys=True, ensure_ascii=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def derive_request_seed(base_seed: int | None, identity: dict[str, Any] | None) -> int | None:
    if base_seed is None:
        return None
    if identity is None:
        raise ValueError("Seeded generation requires a canonical request identity")
    payload = json.dumps(
        {"base_seed": base_seed, "identity": identity},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**31)


def render_chat_prompts(tokenizer: Any, conversations: list[list[dict[str, str]]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        for conversation in conversations
    ]


class HuggingFaceGenerator:
    actual_backend = "hf"

    def __init__(self, model: Any, tokenizer: Any, *, engine_load_seconds: float = 0.0) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.engine_load_seconds = engine_load_seconds

    def generate(
        self,
        conversations: list[list[dict[str, str]]],
        spec: GenerationSpec,
        identities: list[dict[str, Any] | None],
    ) -> list[GeneratedText]:
        import torch

        encoded = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": spec.max_new_tokens,
            "do_sample": spec.do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if spec.do_sample:
            if len(conversations) != 1:
                raise ValueError(
                    "Seeded HF parity generation requires eval_batch_size=1 for per-request RNG"
                )
            generation_kwargs["temperature"] = max(spec.temperature, 1.0e-5)
            generation_kwargs["top_p"] = spec.top_p
            request_seed = derive_request_seed(spec.seed, identities[0])
            if request_seed is not None:
                torch.manual_seed(request_seed)
                torch.cuda.manual_seed_all(request_seed)
        if spec.stop:
            raise ValueError("The HF parity backend does not support string stop sequences")
        with torch.inference_mode():
            output = self.model.generate(**encoded, **generation_kwargs)
        prompt_length = int(encoded["input_ids"].shape[-1])
        token_rows = output[:, prompt_length:]
        contents = self.tokenizer.batch_decode(token_rows, skip_special_tokens=True)
        pad_token_id = self.tokenizer.pad_token_id
        return [
            GeneratedText(
                text=content,
                generated_tokens=int((tokens != pad_token_id).sum().item()),
            )
            for content, tokens in zip(contents, token_rows, strict=True)
        ]


class VLLMGenerator:
    actual_backend = "vllm"

    def __init__(
        self,
        *,
        model_path: Path,
        tokenizer: Any,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        enforce_eager: bool = False,
        model_impl: str = "auto",
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
        shs_backend: str | None = None,
        shs_dispatch_receipt: Path | None = None,
    ) -> None:
        from vllm import LLM

        if shs_backend is not None:
            if model_impl != "transformers":
                raise ValueError("SHS evaluation requires vLLM model_impl=transformers")
            if shs_dispatch_receipt is None:
                raise ValueError("SHS evaluation requires a dispatch receipt path")
            if shs_dispatch_receipt.exists():
                raise FileExistsError(f"Refusing to overwrite SHS dispatch receipt: {shs_dispatch_receipt}")
            shs_dispatch_receipt.parent.mkdir(parents=True, exist_ok=True)
            os.environ["SHS_INFERENCE_MUL_BACKEND"] = shs_backend
            os.environ["SHS_DISPATCH_RECEIPT"] = str(shs_dispatch_receipt.resolve())
        started = time.perf_counter()
        self.tokenizer = tokenizer
        self.shs_backend = shs_backend
        self.shs_dispatch_receipt = shs_dispatch_receipt
        engine_kwargs: dict[str, Any] = {}
        if max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        self.engine = LLM(
            model=str(model_path),
            tokenizer=str(model_path),
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            model_impl=model_impl,
            **engine_kwargs,
        )
        self.engine_load_seconds = time.perf_counter() - started

    def generate(
        self,
        conversations: list[list[dict[str, str]]],
        spec: GenerationSpec,
        identities: list[dict[str, Any] | None],
    ) -> list[GeneratedText]:
        from vllm import SamplingParams

        prompts = render_chat_prompts(self.tokenizer, conversations)
        params = [
            SamplingParams(
                max_tokens=spec.max_new_tokens,
                temperature=spec.temperature if spec.do_sample else 0.0,
                top_p=spec.top_p,
                seed=derive_request_seed(spec.seed, identity),
                stop=list(spec.stop) or None,
            )
            for identity in identities
        ]
        outputs = self.engine.generate(prompts, params, use_tqdm=False)
        if self.shs_backend is not None:
            assert self.shs_dispatch_receipt is not None
            validate_shs_dispatch_receipts(self.shs_dispatch_receipt, self.shs_backend)
        if len(outputs) != len(prompts):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts")
        results: list[GeneratedText] = []
        for output in outputs:
            if len(output.outputs) != 1:
                raise RuntimeError("The evaluator requires exactly one completion per request")
            completion = output.outputs[0]
            results.append(
                GeneratedText(
                    text=completion.text,
                    generated_tokens=len(completion.token_ids),
                    finish_reason=str(completion.finish_reason),
                )
            )
        return results


def generation_spec_receipt(spec: GenerationSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["stop"] = list(spec.stop)
    return payload


def validate_shs_dispatch_receipts(path: Path, requested_backend: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"SHS dispatch receipt was not created: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 3:
        raise RuntimeError(f"Expected 3 SHS projection receipts, found {len(rows)}")
    actual = [str(row.get("backend")) for row in rows]
    if actual != [requested_backend] * 3:
        raise RuntimeError(f"SHS backend mismatch: requested={requested_backend}, actual={actual}")
    if any(bool(row.get("fallback", False)) for row in rows):
        raise RuntimeError("SHS dispatch receipt reported a fallback")
    return rows
