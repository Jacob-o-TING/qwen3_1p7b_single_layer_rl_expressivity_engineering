from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from scripts.audit_shs_weight_sync_checkpoint import tensor_hash


class AuditShsWeightSyncCheckpointTests(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for tensor hashing")
    def test_tensor_hash_supports_bfloat16_and_detects_change(self) -> None:
        left = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        right = left.clone()
        self.assertEqual(tensor_hash(left), tensor_hash(right))
        right[0] = 3.0
        self.assertNotEqual(tensor_hash(left), tensor_hash(right))


if __name__ == "__main__":
    unittest.main()
