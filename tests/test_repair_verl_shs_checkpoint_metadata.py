from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.repair_verl_shs_checkpoint_metadata import AUTO_MAP, repair


class RepairVerlShsCheckpointMetadataTests(unittest.TestCase):
    def test_repair_preserves_original_config_and_writes_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "model_type": "qwen3_shs",
                "auto_map": {"AutoConfig": "shs_hf_model.Qwen3SHSConfig"},
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            receipt = repair(root, root / "receipt.json")

            repaired = json.loads((root / "config.json").read_text(encoding="utf-8"))
            backup = json.loads((root / "config.pre_shs_metadata_repair.json").read_text(encoding="utf-8"))
            self.assertEqual(repaired["auto_map"], AUTO_MAP)
            self.assertEqual(backup, config)
            self.assertTrue((root / "configuration_qwen3_shs.py").is_file())
            self.assertTrue((root / "modeling_qwen3_shs.py").is_file())
            self.assertFalse(receipt["tensor_files_modified"])

    def test_repair_rejects_non_shs_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                repair(root, root / "receipt.json")


if __name__ == "__main__":
    unittest.main()
