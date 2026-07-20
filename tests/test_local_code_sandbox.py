from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from qwen_single_layer_rl.eval.local_code_sandbox import (
    PrivilegeDroppedLocalSandboxBackend,
    _sandbox_environment,
)


class LocalCodeSandboxTests(unittest.TestCase):
    def test_numeric_libraries_are_pinned_to_one_thread(self) -> None:
        environment = _sandbox_environment(Path("/tmp/sample"))

        for variable in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            self.assertEqual(environment[variable], "1")

    def test_environment_keeps_isolated_python_controls(self) -> None:
        environment = _sandbox_environment(Path("/tmp/sample"))

        self.assertEqual(environment["HOME"], str(Path("/tmp/sample")))
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")

    @patch("qwen_single_layer_rl.eval.local_code_sandbox.shutil.rmtree")
    @patch("qwen_single_layer_rl.eval.local_code_sandbox.subprocess.run")
    @patch("qwen_single_layer_rl.eval.local_code_sandbox.os.chmod")
    @unittest.skipUnless(hasattr(os, "chown"), "privilege-drop sandbox requires POSIX chown")
    @patch("qwen_single_layer_rl.eval.local_code_sandbox.os.chown")
    @patch("qwen_single_layer_rl.eval.local_code_sandbox.tempfile.mkdtemp")
    def test_success_result_exposes_evalscope_output_contract(
        self,
        mkdtemp: MagicMock,
        _chown: MagicMock,
        _chmod: MagicMock,
        run: MagicMock,
        _rmtree: MagicMock,
    ) -> None:
        mkdtemp.return_value = "/tmp/qwen_sandbox_test"
        run.return_value = MagicMock(
            returncode=0,
            stdout="TEST_PASSED\n",
            stderr="",
        )
        backend = PrivilegeDroppedLocalSandboxBackend(
            Path("/runner.py"),
            Path("/python"),
            Path("/tmp"),
        )

        with patch.object(Path, "write_text"), patch.object(Path, "chmod"):
            result = backend.execute("print('TEST_PASSED')", 6, "python")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["output"], "TEST_PASSED\n")
        self.assertEqual(result["stdout"], "TEST_PASSED\n")


if __name__ == "__main__":
    unittest.main()
