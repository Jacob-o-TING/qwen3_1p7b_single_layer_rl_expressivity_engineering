from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.eval.generator_backend import (
    GenerationSpec,
    JsonlReceiptWriter,
    derive_request_seed,
    prompt_sha256,
    render_chat_prompts,
    validate_shs_dispatch_receipts,
)
from qwen_single_layer_rl.vllm.custom_ffn_contract import archive_incomplete_dispatch_receipt


class _Tokenizer:
    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        self.last_tokenize = tokenize
        self.last_add_generation_prompt = add_generation_prompt
        return "<user>" + conversation[0]["content"] + "<assistant>"


class GeneratorBackendTests(unittest.TestCase):
    def test_incomplete_dispatch_receipt_is_archived_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "triglu_dispatch_receipts.jsonl"
            receipt.write_text('{"attempt": 1}\n', encoding="utf-8")

            first = archive_incomplete_dispatch_receipt(receipt)
            self.assertEqual(first, receipt.with_name("triglu_dispatch_receipts.attempt-01.jsonl"))
            self.assertFalse(receipt.exists())
            self.assertEqual(first.read_text(encoding="utf-8"), '{"attempt": 1}\n')

            receipt.write_text('{"attempt": 2}\n', encoding="utf-8")
            second = archive_incomplete_dispatch_receipt(receipt)
            self.assertEqual(second, receipt.with_name("triglu_dispatch_receipts.attempt-02.jsonl"))
            self.assertEqual(second.read_text(encoding="utf-8"), '{"attempt": 2}\n')

    def test_chat_prompt_contract_is_shared_and_stable(self) -> None:
        tokenizer = _Tokenizer()
        prompts = render_chat_prompts(tokenizer, [[{"role": "user", "content": "2+2?"}]])

        self.assertEqual(prompts, ["<user>2+2?<assistant>"])
        self.assertFalse(tokenizer.last_tokenize)
        self.assertTrue(tokenizer.last_add_generation_prompt)
        self.assertEqual(len(prompt_sha256(prompts[0])), 64)

    def test_generation_spec_is_hashable_for_batch_signatures(self) -> None:
        spec = GenerationSpec(3072, 0.0, False, 1.0, 20260707, ())
        self.assertEqual({spec}, {spec})

    def test_request_seed_is_identity_stable_and_sample_specific(self) -> None:
        first = {"benchmark": "paper_amc23", "item_id": 4, "sample_id": 0}
        second = {"benchmark": "paper_amc23", "item_id": 4, "sample_id": 1}
        self.assertEqual(derive_request_seed(20260707, first), derive_request_seed(20260707, first))
        self.assertNotEqual(derive_request_seed(20260707, first), derive_request_seed(20260707, second))

    def test_seeded_generation_rejects_missing_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical request identity"):
            derive_request_seed(20260707, None)

    def test_shs_dispatch_receipts_require_three_exact_backends(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dispatch.jsonl"
            path.write_text(
                "".join(json.dumps({"backend": "reference", "fallback": False}) + "\n" for _ in range(3)),
                encoding="utf-8",
            )
            rows = validate_shs_dispatch_receipts(path, "reference")
            self.assertEqual(len(rows), 3)
            path.write_text(
                json.dumps({"backend": "reference"}) + "\n" + json.dumps({"backend": "triton"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "Expected 3"):
                validate_shs_dispatch_receipts(path, "reference")

    def test_receipts_are_append_only_and_count_completions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.jsonl"
            writer = JsonlReceiptWriter(path)
            writer.append({"event": "engine_loaded", "actual_backend": "vllm"})
            writer.append({"event": "generation_completed", "prompt_sha256": "abc"})
            writer.append({"event": "generation_completed", "prompt_sha256": "def"})
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["event"] for row in rows], [
            "engine_loaded", "generation_completed", "generation_completed"
        ])
        self.assertEqual(rows[1]["completed_count"], 1)
        self.assertEqual(rows[2]["completed_count"], 2)


if __name__ == "__main__":
    unittest.main()
