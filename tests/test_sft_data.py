from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.sft.data import (
    IGNORE_INDEX,
    EncodedExample,
    build_packed_dataset,
    encode_supervised_example,
    pack_examples,
)


class FakeTokenizer:
    name_or_path = "fake-qwen-tokenizer"
    chat_template = "fake"
    eos_token_id = 2
    pad_token_id = 0

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [10 + ord(character) % 83 for character in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        del tokenize
        rendered = "".join(f"<{item['role']}>{item['content']}" for item in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return self._encode(rendered)


class SFTDataTests(unittest.TestCase):
    def test_prompt_is_masked_and_solution_is_supervised(self) -> None:
        tokenizer = FakeTokenizer()
        record = {
            "problem": "What is 1+1?",
            "solution": "Reasoning, then 2.",
            "messages": [
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": "Reasoning, then 2."},
            ],
        }
        encoded = encode_supervised_example(tokenizer, record, source_index=7, max_length=512)
        first_supervised = next(index for index, label in enumerate(encoded.labels) if label != IGNORE_INDEX)
        prompt_ids = tokenizer.apply_chat_template(
            record["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        self.assertEqual(first_supervised, len(prompt_ids))
        self.assertTrue(all(label == IGNORE_INDEX for label in encoded.labels[:first_supervised]))
        self.assertEqual(encoded.source_index, 7)
        self.assertEqual(encoded.input_ids[-1], tokenizer.eos_token_id)

    def test_final_answer_is_not_accepted_as_full_solution(self) -> None:
        with self.assertRaisesRegex(ValueError, "full solution"):
            encode_supervised_example(
                FakeTokenizer(),
                {"problem": "Problem", "answer": "42"},
                source_index=0,
                max_length=128,
            )

    def test_packing_preserves_source_order_and_masks_padding(self) -> None:
        examples = [
            EncodedExample((10, 11), (IGNORE_INDEX, 11), 4, 0),
            EncodedExample((12, 13, 14), (IGNORE_INDEX, 13, 14), 9, 0),
            EncodedExample((15, 16, 17, 18), (IGNORE_INDEX, 16, 17, 18), 12, 0),
        ]
        items, manifest = pack_examples(examples, max_length=6, pad_token_id=0)
        self.assertEqual([item["source_indices"] for item in items], [(4, 9), (12,)])
        self.assertEqual(items[1]["attention_mask"], (1, 1, 1, 1, 0, 0))
        self.assertEqual(items[1]["labels"][-2:], (IGNORE_INDEX, IGNORE_INDEX))
        self.assertEqual(manifest["example_count"], 3)
        self.assertEqual(manifest["packed_sequence_count"], 2)

    def test_build_dataset_uses_full_solution_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(
                '{"problem":"P","solution":"Detailed solution","answer":"A"}\n',
                encoding="utf-8",
            )
            dataset, manifest = build_packed_dataset(FakeTokenizer(), path, max_length=128)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(manifest["example_count"], 1)
        self.assertEqual(manifest["label_policy"], "assistant_only_ignore_index_-100")

    def test_overlong_prompt_is_deterministically_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"problem":"' + ("P" * 100) + '","solution":"S"}',
                        '{"problem":"short","solution":"full solution"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            dataset, manifest = build_packed_dataset(FakeTokenizer(), path, max_length=64)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(manifest["dropped_overlong_prompt_count"], 1)
        self.assertEqual(manifest["dropped_overlong_prompt_indices"], [0])


if __name__ == "__main__":
    unittest.main()
