"""
tools/sandbox_tool.py
=====================
Executes arbitrary Python code inside a restricted subprocess sandbox
and captures stdout, stderr, and the return code.

Imports
-------
    stdlib  : logging, subprocess, sys, textwrap, time, tempfile,
              os, signal, resource (POSIX only)
    internal: interfaces.base_tool.BaseTool
              contracts.tool_contracts.{ToolRequest, ToolResponse,
                  ToolStatus, ExecutionResult}
              contracts.shared_context.SharedContext

Inputs  : ToolRequest
    payload keys:
        code        (str, required)   — Python source to execute
        timeout     (float, default=10.0) — max wall-clock seconds
        allow_net   (bool, default=False) — reserved for future firewall
        stdin       (str, default="")    — optional stdin content

Outputs : ToolResponse
    data: ExecutionResult
        stdout      : captured standard output
        stderr      : captured standard error
        return_code : process exit code  (0 = success)
        timed_out   : True if process was killed due to timeout
        exec_ms     : wall-clock duration in milliseconds

Exceptions handled
------------------
    ValueError   — missing/blank code field, bad timeout value
    TypeError    — payload not a dict
    TimeoutError — re-raised from subprocess timeout to trigger BaseTool retry
    OSError      — subprocess launch failure (retried)
    Exception    — catch-all (retried)

Dependencies
------------
    stdlib only — no third-party packages required.

Security notes
--------------
    • Code runs in a *separate* interpreter process (subprocess).
    • Memory capped via resource.RLIMIT_AS on POSIX systems.
    • Network is not explicitly blocked here; set allow_net=False and
      add iptables/seccomp rules at the host level for production.
    • Temp file is always cleaned up in a finally block.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from interfaces.base_tool import BaseTool
from contracts.tool_contracts import (
    ExecutionResult,
    ToolRequest,
    ToolResponse,
    ToolStatus,
)
from contracts.shared_context import SharedContext

logger = logging.getLogger(__name__)

# Maximum bytes returned from stdout / stderr to prevent context explosion
_MAX_OUTPUT_BYTES = 64 * 1024  # 64 KB

# Memory cap for the child process (256 MB) — POSIX only
_MEM_LIMIT_BYTES = 256 * 1024 * 1024


def _posix_preexec() -> None:
    """Called in the child process before exec on POSIX systems."""
    try:
        import resource  # noqa: PLC0415

        resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))
    except Exception:  # noqa: BLE001
        pass  # non-critical; best-effort


class SandboxTool(BaseTool):
    """
    Secure Python sandbox using subprocess isolation.

    The user's code is written to a temp file and executed in a fresh
    interpreter, ensuring no shared global state between invocations.
    """

    TOOL_NAME      : str   = "sandbox"
    VERSION        : str   = "1.0.0"
    MAX_RETRIES    : int   = 2          # fewer retries for compute tasks
    TIMEOUT_SECONDS: float = 10.0

    # ── validation ────────────────────────────────────────────────────── #

    def validate(self, request: ToolRequest) -> None:
        if not isinstance(request.payload, dict):
            raise TypeError(f"payload must be dict, got {type(request.payload).__name__}")

        code = request.payload.get("code", "")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("payload.code must be a non-empty string")

        timeout = request.payload.get("timeout", self.TIMEOUT_SECONDS)
        if not isinstance(timeout, float | int) or float(timeout) <= 0:
            raise ValueError("payload.timeout must be a positive number")

    # ── execution ─────────────────────────────────────────────────────── #

    def execute(self, request: ToolRequest, context: SharedContext) -> ToolResponse:
        t0      = time.monotonic()
        code    = textwrap.dedent(request.payload["code"])
        timeout = float(request.payload.get("timeout", self.TIMEOUT_SECONDS))
        stdin   = request.payload.get("stdin", "")

        tmp_path: Path | None = None
        try:
            # Write code to a temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as fh:
                fh.write(code)
                tmp_path = Path(fh.name)

            preexec = _posix_preexec if sys.platform != "win32" else None

            proc = subprocess.run(  # noqa: S603
                [sys.executable, str(tmp_path)],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=preexec,
            )

            stdout = proc.stdout[:_MAX_OUTPUT_BYTES]
            stderr = proc.stderr[:_MAX_OUTPUT_BYTES]
            exec_ms = (time.monotonic() - t0) * 1000

            logger.debug(
                "[%s] rc=%d stdout_len=%d stderr_len=%d exec_ms=%.1f",
                self.TOOL_NAME,
                proc.returncode,
                len(stdout),
                len(stderr),
                exec_ms,
            )

            result = ExecutionResult(
                stdout     =stdout,
                stderr     =stderr,
                return_code=proc.returncode,
                timed_out  =False,
                exec_ms    =exec_ms,
            )
            return ToolResponse.success(
                request_id =request.request_id,
                tool_name  =self.TOOL_NAME,
                data       =result,
                duration_ms=exec_ms,
            )

        except subprocess.TimeoutExpired:
            exec_ms = (time.monotonic() - t0) * 1000
            logger.warning("[%s] execution timed out after %.1f s", self.TOOL_NAME, timeout)
            result = ExecutionResult(
                stdout     ="",
                stderr     =f"Execution timed out after {timeout:.1f}s",
                return_code=-1,
                timed_out  =True,
                exec_ms    =exec_ms,
            )
            # Return TIMEOUT status — BaseTool will NOT retry a returned response
            return ToolResponse(
                request_id=request.request_id,
                tool_name =self.TOOL_NAME,
                status    =ToolStatus.TIMEOUT,
                data      =result,
                error     =f"Timed out after {timeout:.1f}s",
                duration_ms=exec_ms,
            )

        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
