from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _sandbox_environment(home: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    }


class PrivilegeDroppedLocalSandboxBackend:
    def __init__(self, runner: Path, python: Path, scratch_root: Path) -> None:
        self.runner = runner.resolve()
        self.python = python.resolve()
        self.scratch_root = scratch_root.resolve()
        self._ready = False

    def start(self) -> None:
        if os.geteuid() != 0:
            raise RuntimeError("local code sandbox launcher requires root only to drop privileges")
        for command in (Path("/usr/bin/setpriv"), self.runner, self.python):
            if not command.exists():
                raise RuntimeError(f"local code sandbox dependency missing: {command}")
        if not (stat.S_IMODE(Path("/root").stat().st_mode) & stat.S_IXOTH):
            raise RuntimeError("/root must be traverse-only for the unprivileged Python interpreter")
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.scratch_root, 0o711)
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def execute(self, code: str | list[str], timeout: int, language: str) -> dict[str, Any]:
        if language.lower() != "python":
            return {"status": "error", "error": f"unsupported language: {language}"}
        if isinstance(code, list):
            code = "\n".join(code)
        directory = Path(tempfile.mkdtemp(prefix="sample_", dir=self.scratch_root))
        try:
            os.chown(directory, 65534, 65534)
            os.chmod(directory, 0o700)
            payload = directory / "payload.py"
            payload.write_text(code, encoding="utf-8")
            os.chown(payload, 65534, 65534)
            os.chmod(payload, 0o600)
            environment = _sandbox_environment(directory)
            command = [
                "/usr/bin/setpriv",
                "--reuid=nobody",
                "--regid=nogroup",
                "--clear-groups",
                "--no-new-privs",
                str(self.python),
                "-I",
                "-B",
                str(self.runner),
                str(max(1, int(timeout))),
                str(payload),
            ]
            try:
                result = subprocess.run(
                    command,
                    cwd=directory,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=max(1, int(timeout)) + 3,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return {
                    "status": "timeout",
                    "error": "Code execution timed out.",
                    "stdout": (exc.stdout or "")[-65536:],
                    "stderr": (exc.stderr or "")[-65536:],
                }
            status = "success" if result.returncode == 0 else "error"
            return {
                "status": status,
                "exit_code": result.returncode,
                # EvalScope's sandbox contract reads the program's stdout from
                # `output`; retain `stdout` as diagnostic evidence as well.
                "output": result.stdout[-65536:],
                "stdout": result.stdout[-65536:],
                "stderr": result.stderr[-65536:],
                "isolation": "uid65534+no_new_privs+rlimits+seccomp_no_network_no_exec",
            }
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def stop(self) -> None:
        self._ready = False


def install_evalscope_local_code_sandbox(*, runner: Path, python: Path, scratch_root: Path) -> None:
    from evalscope.config import TaskConfig
    from evalscope.api.mixin.sandbox_mixin import SandboxMixin

    if getattr(SandboxMixin, "_qwen_local_sandbox_installed", False):
        return

    def get_backend(instance: Any) -> PrivilegeDroppedLocalSandboxBackend:
        if instance._backend is None:
            instance._backend = PrivilegeDroppedLocalSandboxBackend(runner, python, scratch_root)
        return instance._backend

    SandboxMixin._get_backend = get_backend
    SandboxMixin._qwen_local_sandbox_installed = True

    if not getattr(TaskConfig, "_qwen_local_sandbox_installed", False):
        original_init = TaskConfig._init_default_sandbox_config

        def init_local_sandbox_config(instance: Any) -> None:
            try:
                original_init(instance)
            except ImportError as exc:
                if "ms_enclave" not in str(exc):
                    raise
                sandbox = getattr(instance, "sandbox", None)
                if sandbox is None or not sandbox.enabled:
                    raise

        TaskConfig._init_default_sandbox_config = init_local_sandbox_config
        TaskConfig._qwen_local_sandbox_installed = True
