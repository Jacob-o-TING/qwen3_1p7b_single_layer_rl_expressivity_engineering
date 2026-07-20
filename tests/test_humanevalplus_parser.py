from __future__ import annotations

import unittest

from qwen_single_layer_rl.eval.humanevalplus_parser import (
    parse_humanevalplus_prediction,
)


class HumanEvalPlusParserTests(unittest.TestCase):
    def test_clean_function_is_byte_exact(self) -> None:
        raw = "def answer(x):\n    return x + 1\n"
        parsed, receipt = parse_humanevalplus_prediction(raw)
        self.assertEqual(parsed, raw)
        self.assertFalse(receipt.changed)
        self.assertEqual(receipt.strategy, "unchanged")

    def test_indented_completion_body_is_preserved(self) -> None:
        raw = "    result = x + 1\n    return result\n"
        parsed, receipt = parse_humanevalplus_prediction(raw)
        self.assertEqual(parsed, raw)
        self.assertFalse(receipt.changed)

    def test_hebrew_prefix_on_separate_lines_is_removed(self) -> None:
        raw = "תשובתך\nתשובתך: def answer(x):\n    return x + 1\n"
        parsed, receipt = parse_humanevalplus_prediction(raw)
        self.assertEqual(parsed, "def answer(x):\n    return x + 1\n")
        self.assertEqual(receipt.strategy, "strip_prefix_to_python_declaration")

    def test_markdown_uses_first_code_block(self) -> None:
        raw = "Here is the code:\n```python\ndef answer(x):\n    return x\n```\nThanks"
        parsed, receipt = parse_humanevalplus_prediction(raw)
        self.assertEqual(parsed, "def answer(x):\n    return x\n")
        self.assertEqual(receipt.strategy, "first_fenced_code_block")

    def test_complete_think_block_is_removed(self) -> None:
        raw = "<think>reasoning</think>\ndef answer(x):\n    return x\n"
        parsed, receipt = parse_humanevalplus_prediction(raw)
        self.assertEqual(parsed, "def answer(x):\n    return x\n")
        self.assertTrue(receipt.changed)

    def test_no_code_anchor_fails_closed(self) -> None:
        raw = "תשובתך\nI cannot solve this."
        parsed, receipt = parse_humanevalplus_prediction(raw)
        self.assertEqual(parsed, raw)
        self.assertFalse(receipt.changed)

    def test_strips_non_code_prefix_before_indented_function_body(self) -> None:
        raw = "answer follows\n    if not numbers:\n        return []\n    return numbers\n"
        parsed, receipt = parse_humanevalplus_prediction(raw)
        self.assertEqual(
            parsed,
            "    if not numbers:\n        return []\n    return numbers\n",
        )
        self.assertEqual(
            receipt.strategy, "strip_prefix_to_indented_function_body"
        )
        self.assertTrue(receipt.changed)

    def test_non_code_prefix_without_indented_body_fails_closed(self) -> None:
        raw = "answer follows\nthere is no code here\n"
        parsed, receipt = parse_humanevalplus_prediction(raw)
        self.assertEqual(parsed, raw)
        self.assertEqual(receipt.strategy, "unchanged")
        self.assertFalse(receipt.changed)


if __name__ == "__main__":
    unittest.main()
