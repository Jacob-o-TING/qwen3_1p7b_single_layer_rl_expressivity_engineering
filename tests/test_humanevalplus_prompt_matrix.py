from __future__ import annotations

import unittest

from qwen_single_layer_rl.eval.humanevalplus_prompt_matrix import (
    EVALSCOPE_INSTRUCTION,
    classify_generation,
    decide_canary_escalation,
    render_prompt,
    request_seed,
    select_tasks,
    summarize_cell,
)


class _Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        return f"<user>{messages[0]['content']}<assistant>"


class HumanEvalPlusPromptMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n    \"\"\"Return a+b.\"\"\"\n",
            "entry_point": "add",
            "test": "",
            "source_index": 0,
        }

    def test_rendering_changes_only_the_approved_interface(self) -> None:
        tokenizer = _Tokenizer()
        control = render_prompt(tokenizer, "evalscope_chat_instruction_control", self.task)
        raw = render_prompt(tokenizer, "evalscope_raw_instruction_nochat", self.task)
        canonical = render_prompt(tokenizer, "canonical_completion_nochat", self.task)
        self.assertEqual(raw, EVALSCOPE_INSTRUCTION + self.task["prompt"])
        self.assertEqual(canonical, self.task["prompt"])
        self.assertEqual(control, f"<user>{raw}<assistant>")
        self.assertFalse(tokenizer.tokenize)
        self.assertTrue(tokenizer.add_generation_prompt)

    def test_canary_selection_is_stable_and_order_independent(self) -> None:
        tasks = [{**self.task, "task_id": f"HumanEval/{index}", "source_index": index} for index in range(10)]
        first = select_tasks(tasks, seed=20260707, count=4)
        second = select_tasks(reversed(tasks), seed=20260707, count=4)
        self.assertEqual(first, second)
        self.assertEqual([row["ledger_index"] for row in first], list(range(4)))

    def test_request_seed_is_cell_independent_and_task_specific(self) -> None:
        self.assertEqual(request_seed(7, "HumanEval/0"), request_seed(7, "HumanEval/0"))
        self.assertNotEqual(request_seed(7, "HumanEval/0"), request_seed(7, "HumanEval/1"))

    def test_generation_classification_detects_collapse_and_valid_body(self) -> None:
        loop = classify_generation(
            raw="\u05ea\u05e9\u05d5\u05d1\u05ea\n" * 8,
            parsed="\u05ea\u05e9\u05d5\u05d1\u05ea\n" * 8,
            canonical_prompt=self.task["prompt"],
            finish_reason="length",
        )
        self.assertTrue(loop["hebrew_loop"])
        self.assertTrue(loop["cap_hit"])
        body = classify_generation(
            raw="    return a + b\n",
            parsed="    return a + b\n",
            canonical_prompt=self.task["prompt"],
            finish_reason="stop",
        )
        self.assertTrue(body["syntax_valid_completion"])
        self.assertFalse(body["collapse_loop"])

    def test_escalation_requires_both_pass_gain_and_loop_reduction(self) -> None:
        summaries = {
            "evalscope_chat_instruction_control": {"passed": 1, "collapse_loops": 28, "syntax_valid_completions": 2},
            "evalscope_raw_instruction_nochat": {"passed": 5, "collapse_loops": 20, "syntax_valid_completions": 8},
            "canonical_completion_nochat": {"passed": 8, "collapse_loops": 2, "syntax_valid_completions": 25},
        }
        decision = decide_canary_escalation(summaries)
        self.assertEqual(decision["winning_cell"], "canonical_completion_nochat")
        self.assertTrue(decision["escalate_to_full_untuned_base"])
        summaries["canonical_completion_nochat"]["collapse_loops"] = 27
        self.assertFalse(decide_canary_escalation(summaries)["escalate_to_full_untuned_base"])

    def test_cell_summary_preserves_finish_and_parser_distributions(self) -> None:
        row = {
            "generated_tokens": 12,
            "finish_reason": "stop",
            "parser": {"strategy": "unchanged"},
            "execution_status": "success",
            "passed": True,
            "classification": {
                "contains_hebrew": False,
                "hebrew_loop": False,
                "english_fragment_loop": False,
                "collapse_loop": False,
                "syntax_valid_completion": True,
                "cap_hit": False,
            },
        }
        summary = summarize_cell([row])
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["finish_reasons"], {"stop": 1})
        self.assertEqual(summary["parser_strategies"], {"unchanged": 1})
        self.assertEqual(summary["execution_timeouts"], 0)

    def test_sigxcpu_is_classified_as_sandbox_timeout(self) -> None:
        row = {
            "generated_tokens": 20,
            "finish_reason": "stop",
            "parser": {"strategy": "unchanged"},
            "execution_status": "error",
            "execution_exit_code": -24,
            "passed": False,
            "classification": {
                "contains_hebrew": False,
                "hebrew_loop": False,
                "english_fragment_loop": False,
                "collapse_loop": False,
                "syntax_valid_completion": True,
                "cap_hit": False,
            },
        }
        self.assertEqual(summarize_cell([row])["execution_timeouts"], 1)


if __name__ == "__main__":
    unittest.main()
