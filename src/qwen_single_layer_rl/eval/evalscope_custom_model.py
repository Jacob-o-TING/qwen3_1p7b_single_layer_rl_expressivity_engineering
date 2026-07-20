from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .checkpoint_loader import load_sft_checkpoint_for_inference
from .generator_backend import (
    GenerationSpec,
    HuggingFaceGenerator,
    JsonlReceiptWriter,
    VLLMGenerator,
    derive_request_seed,
    generation_spec_receipt,
    prompt_sha256,
    render_chat_prompts,
)


def canonical_request_identity(
    explicit_identity: dict[str, Any] | None,
    *,
    prompt: str,
    namespace: str,
) -> dict[str, Any]:
    if explicit_identity is not None:
        return dict(explicit_identity)
    return {
        "source": "evalscope_rendered_prompt",
        "namespace": namespace,
        "prompt_sha256": prompt_sha256(prompt),
    }


@dataclass
class _PendingRequest:
    payload: Any
    signature: tuple[Any, ...]
    max_batch_size: int
    completed: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


class _SynchronousMicroBatcher:
    def __init__(
        self,
        process_batch: Callable[[list[Any]], list[Any]],
        *,
        wait_seconds: float = 0.010,
    ) -> None:
        self._process_batch = process_batch
        self._wait_seconds = wait_seconds
        self._condition = threading.Condition()
        self._pending: list[_PendingRequest] = []
        self._worker = threading.Thread(target=self._run, name="qwen-eval-batcher", daemon=True)
        self._worker.start()

    def submit(self, payload: Any, *, signature: tuple[Any, ...], max_batch_size: int) -> Any:
        if max_batch_size <= 1:
            return self._process_batch([payload])[0]
        request = _PendingRequest(
            payload=payload,
            signature=signature,
            max_batch_size=max_batch_size,
        )
        with self._condition:
            self._pending.append(request)
            self._condition.notify_all()
        request.completed.wait()
        if request.error is not None:
            raise request.error
        return request.result

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending:
                    self._condition.wait()
                first = self._pending.pop(0)
                batch = [first]
                deadline = time.monotonic() + self._wait_seconds
                while len(batch) < first.max_batch_size:
                    match_index = next(
                        (
                            index
                            for index, request in enumerate(self._pending)
                            if request.signature == first.signature
                        ),
                        None,
                    )
                    if match_index is not None:
                        batch.append(self._pending.pop(match_index))
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)
            try:
                outputs = self._process_batch([request.payload for request in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"Batch processor returned {len(outputs)} outputs for {len(batch)} requests"
                    )
                for request, output in zip(batch, outputs, strict=True):
                    request.result = output
            except BaseException as exc:
                for request in batch:
                    request.error = exc
            finally:
                for request in batch:
                    request.completed.set()


def register_evalscope_model() -> type:
    try:
        from evalscope.api.messages import ChatMessage
        from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput
        from evalscope.api.registry import register_model_api
        from evalscope.api.tool import ToolChoice, ToolInfo
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "EvalScope is not installed. Use the pinned evaluation environment before final evaluation."
        ) from exc

    @register_model_api(name="qwen_single_layer_sft")
    class QwenSingleLayerSFTModel(ModelAPI):
        def __init__(
            self,
            model_name: str,
            base_url: str | None = None,
            api_key: str | None = None,
            config: GenerateConfig = GenerateConfig(),
            **model_args: dict[str, Any],
        ) -> None:
            super().__init__(model_name, base_url, api_key, config)
            self.model_args = model_args
            self.identity_namespace = str(
                model_args.get("identity_namespace", "evalscope_unspecified")
            )
            requested_backend = str(model_args.get("backend", "hf"))
            if requested_backend not in {"hf", "vllm"}:
                raise ValueError(f"Unsupported evaluator backend: {requested_backend}")
            source_load_started = time.perf_counter()
            checkpoint_arg = model_args.get("checkpoint_dir")
            if requested_backend == "vllm":
                from transformers import AutoTokenizer

                from qwen_single_layer_rl.config import load_config

                if not bool(model_args.get("base_model_only", False)):
                    raise ValueError("The first vLLM evaluator slice supports native base_model_only only")
                model_path = model_args.get("model_path")
                if not model_path:
                    raise ValueError("The vLLM evaluator backend requires model_path")
                self.project_config = load_config(Path(str(model_args["config_path"])))
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path), trust_remote_code=True
                )
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                model = None
            else:
                model, self.tokenizer, self.project_config = load_sft_checkpoint_for_inference(
                    config_path=Path(str(model_args["config_path"])),
                    checkpoint_dir=Path(str(checkpoint_arg)) if checkpoint_arg else None,
                    model_path=(
                        Path(str(model_args["model_path"])) if model_args.get("model_path") else None
                    ),
                    device=str(model_args.get("device", "cuda")),
                    base_model_only=bool(model_args.get("base_model_only", False)),
                )
            self.tokenizer.padding_side = "left"
            receipt_arg = model_args.get("receipt_jsonl")
            self._receipts = JsonlReceiptWriter(Path(str(receipt_arg)) if receipt_arg else None)
            if requested_backend == "hf":
                assert model is not None
                self.model = model
                self._generator = HuggingFaceGenerator(
                    model,
                    self.tokenizer,
                    engine_load_seconds=time.perf_counter() - source_load_started,
                )
            else:
                self.model = None
                self._generator = VLLMGenerator(
                    model_path=Path(str(model_path)),
                    tokenizer=self.tokenizer,
                    tensor_parallel_size=int(model_args.get("tensor_parallel_size", 1)),
                    gpu_memory_utilization=float(model_args.get("gpu_memory_utilization", 0.85)),
                    enforce_eager=bool(model_args.get("enforce_eager", False)),
                    model_impl=str(model_args.get("vllm_model_impl", "auto")),
                    max_num_seqs=(
                        int(model_args["max_num_seqs"])
                        if model_args.get("max_num_seqs") is not None
                        else None
                    ),
                    max_num_batched_tokens=(
                        int(model_args["max_num_batched_tokens"])
                        if model_args.get("max_num_batched_tokens") is not None
                        else None
                    ),
                    shs_backend=(
                        str(model_args["shs_backend"]) if model_args.get("shs_backend") else None
                    ),
                    shs_dispatch_receipt=(
                        Path(str(model_args["shs_dispatch_receipt"]))
                        if model_args.get("shs_dispatch_receipt")
                        else None
                    ),
                )
            if self._generator.actual_backend != requested_backend:
                raise RuntimeError(
                    f"Backend fallback is forbidden: requested={requested_backend} "
                    f"actual={self._generator.actual_backend}"
                )
            self._receipts.append(
                {
                    "event": "engine_loaded",
                    "requested_backend": requested_backend,
                    "actual_backend": self._generator.actual_backend,
                    "engine_load_seconds": self._generator.engine_load_seconds,
                }
            )
            self._batcher = _SynchronousMicroBatcher(
                self._generate_batch,
                wait_seconds=float(model_args.get("microbatch_wait_seconds", 0.010)),
            )

        @staticmethod
        def _messages(input_messages: list[ChatMessage]) -> list[dict[str, str]]:
            messages: list[dict[str, str]] = []
            for message in input_messages:
                role = str(getattr(message, "role", "user"))
                content = getattr(message, "content", str(message))
                if not isinstance(content, str):
                    content = str(content)
                messages.append({"role": role, "content": content})
            return messages

        @staticmethod
        def _identity(input_messages: list[ChatMessage]) -> dict[str, Any] | None:
            for message in reversed(input_messages):
                metadata = getattr(message, "metadata", None)
                if isinstance(metadata, dict) and isinstance(metadata.get("eval_identity"), dict):
                    return dict(metadata["eval_identity"])
            return None

        @staticmethod
        def _generation_values(config: GenerateConfig) -> GenerationSpec:
            max_new_tokens = int(
                getattr(config, "max_tokens", None)
                or getattr(config, "max_new_tokens", None)
                or 3072
            )
            temperature = float(getattr(config, "temperature", 0.0) or 0.0)
            do_sample = bool(getattr(config, "do_sample", temperature > 0.0))
            top_p = float(getattr(config, "top_p", 1.0) or 1.0)
            seed_value = getattr(config, "seed", None)
            stop_value = getattr(config, "stop_seqs", None)
            if isinstance(stop_value, str):
                stop = (stop_value,)
            else:
                stop = tuple(str(value) for value in (stop_value or ()))
            return GenerationSpec(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                seed=int(seed_value) if seed_value is not None else None,
                stop=stop,
            )

        def _generate_batch(
            self,
            requests: list[tuple[list[ChatMessage], GenerateConfig]],
        ) -> list[ModelOutput]:
            if not requests:
                return []
            signatures = {self._generation_values(config) for _, config in requests}
            if len(signatures) != 1:
                raise ValueError("A generation micro-batch must use one decoding configuration")
            spec = signatures.pop()
            conversations = [self._messages(input_messages) for input_messages, _ in requests]
            prompts = render_chat_prompts(self.tokenizer, conversations)
            identities = [
                canonical_request_identity(
                    self._identity(input_messages),
                    prompt=prompt,
                    namespace=self.identity_namespace,
                )
                for prompt, (input_messages, _) in zip(prompts, requests, strict=True)
            ]
            started = time.perf_counter()
            generated = self._generator.generate(conversations, spec, identities)
            elapsed = time.perf_counter() - started
            for prompt, identity, result in zip(prompts, identities, generated, strict=True):
                self._receipts.append(
                    {
                        "event": "generation_completed",
                        "requested_backend": str(self.model_args.get("backend", "hf")),
                        "actual_backend": self._generator.actual_backend,
                        "prompt_sha256": prompt_sha256(prompt),
                        "identity": identity,
                        "request_seed": derive_request_seed(spec.seed, identity),
                        "generated_tokens": result.generated_tokens,
                        "finish_reason": result.finish_reason,
                        "batch_size": len(requests),
                        "batch_elapsed_seconds": elapsed,
                        "generation": generation_spec_receipt(spec),
                    }
                )
            return [
                ModelOutput.from_content(model=self.model_name, content=result.text)
                for result in generated
            ]

        def generate(
            self,
            input: list[ChatMessage],
            tools: list[ToolInfo],
            tool_choice: ToolChoice,
            config: GenerateConfig,
        ) -> ModelOutput:
            del tools, tool_choice
            generation_values = self._generation_values(config)
            max_batch_size = max(1, int(getattr(config, "batch_size", 1) or 1))
            return self._batcher.submit(
                (input, config),
                signature=generation_values,
                max_batch_size=max_batch_size,
            )

    return QwenSingleLayerSFTModel
