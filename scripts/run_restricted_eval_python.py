from __future__ import annotations

import ctypes
import errno
import os
import resource
import runpy
import sys
from pathlib import Path


DENIED_SYSCALLS = (
    "accept",
    "accept4",
    "bind",
    "bpf",
    "connect",
    "execve",
    "execveat",
    "keyctl",
    "kexec_load",
    "listen",
    "mount",
    "open_by_handle_at",
    "ptrace",
    "reboot",
    "sendto",
    "setns",
    "socket",
    "socketpair",
    "umount2",
    "unshare",
)


def install_seccomp_filter() -> None:
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]

    allow = 0x7FFF0000
    errno_action = 0x00050000 | errno.EPERM
    context = lib.seccomp_init(allow)
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        for name in DENIED_SYSCALLS:
            syscall = lib.seccomp_syscall_resolve_name(name.encode("ascii"))
            if syscall < 0:
                continue
            result = lib.seccomp_rule_add(context, errno_action, syscall, 0)
            if result != 0:
                raise RuntimeError(f"seccomp_rule_add failed for {name}: {result}")
        result = lib.seccomp_load(context)
        if result != 0:
            raise RuntimeError(f"seccomp_load failed: {result}")
    finally:
        lib.seccomp_release(context)


def apply_resource_limits(timeout: int) -> None:
    limits = {
        resource.RLIMIT_AS: (1536 << 20, 1536 << 20),
        resource.RLIMIT_CORE: (0, 0),
        resource.RLIMIT_CPU: (max(1, timeout), max(1, timeout + 1)),
        resource.RLIMIT_FSIZE: (16 << 20, 16 << 20),
        resource.RLIMIT_NOFILE: (64, 64),
        resource.RLIMIT_NPROC: (32, 32),
    }
    for target, value in limits.items():
        resource.setrlimit(target, value)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_restricted_eval_python.py TIMEOUT CODE.py")
    timeout = int(sys.argv[1])
    code_path = Path(sys.argv[2]).resolve()
    if os.getuid() == 0 or os.geteuid() == 0:
        raise RuntimeError("restricted evaluator must not run as root")
    if not code_path.is_file() or code_path.parent != Path.cwd().resolve():
        raise RuntimeError("code payload must be a file in the isolated working directory")
    apply_resource_limits(timeout)
    install_seccomp_filter()
    runpy.run_path(str(code_path), run_name="__main__")


if __name__ == "__main__":
    main()
