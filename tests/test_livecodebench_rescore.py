from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rescore_livecodebench_sandbox_contract.py"
SPEC = importlib.util.spec_from_file_location("rescore_livecodebench_sandbox_contract", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LiveCodeBenchRescoreTests(unittest.TestCase):
    def test_prediction_loader_uses_assistant_message_not_response_object_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = "```python\nprint(42)\n```"
            for shard_index in range(6):
                path = (
                    root
                    / "shards"
                    / f"shard_{shard_index:02d}"
                    / "main"
                    / "timestamp"
                    / "predictions"
                    / "model"
                    / "live_code_bench_release_latest.jsonl"
                )
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "index": shard_index,
                            "model_output": {"choices": [{"message": {"content": expected}}]},
                            "messages": [
                                {"role": "user", "content": f"question-{shard_index}"},
                                {"role": "assistant", "content": expected},
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

            rows, sources = MODULE.load_predictions(root)

        self.assertEqual(len(rows), 6)
        self.assertEqual(len(sources), 6)
        self.assertTrue(all(row["model_output"] == expected for row in rows))


if __name__ == "__main__":
    unittest.main()
