# interfaces/base_tool.py (ORIGINAL)
"""
Module: interfaces/base_tool.py
=================================
Purpose:
    Defines the abstract BaseTool interface that every tool in the
    multi-agent pipeline must implement. Tools are discrete, stateless
    callables with built-in retry, validation, and failure handling.

    Agents invoke tools through this interface — never calling external
    APIs, databases, or services directly.

Input Datatypes:
    ToolRequest — from contracts/tool_contracts.py

Output Datatypes:
    ToolResponse — from contracts/tool_contracts.py
    ToolFailure  — from contracts/tool_contracts.py

Dependencies:
    - contracts/tool_contracts.py
    - Python 3.11+
    - asyncio (stdlib)
    - abc (stdlib)
    - logging (stdlib)
    - time (stdlib)

SOLID Principle:
    Single Responsibility — this file ONLY defines the tool execution interface.
    Open/Closed — new tools extend BaseTool without modifying it.
    Liskov Substitution — all BaseTool subclasses are interchangeable
    by the agent layer.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

try:
    from contracts.tool_contracts import (
        RetryMetadata,
        RetryStrategy,
        ToolFailure,
        ToolRequest,
        ToolResponse,
        ToolStatus,
    )
except ImportError:
    RetryMetadata = None  # type: ignore[assignment,misc]
    RetryStrategy = None  # type: ignore[assignment,misc]
    ToolFailure = None  # type: ignore[assignment,misc]
    ToolRequest = None  # type: ignore[assignment,misc]
    ToolResponse = None  # type: ignore[assignment,misc]
    ToolStatus = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ToolValidationError(Exception):
    """Raised when a ToolRequest fails input validation."""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name


class ToolExecutionError(Exception):
    """Raised by a concrete tool when execution fails unrecoverably."""

    def __init__(
        self,
        tool_name: str,
        message: str,
        is_retryable: bool = True,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.is_retryable = is_retryable
        self.original_exception = original_exception


# ---------------------------------------------------------------------------
# Abstract BaseTool
# ---------------------------------------------------------------------------


class BaseTool(ABC):
    """
    Abstract base class for all pipeline tools.

    Lifecycle (called by the agent or tool runner)
    -----------------------------------------------
    1. tool.validate_input(request)   — pre-flight guard
    2. response = await tool.execute(request)  — core async logic
       (retry loop managed by safe_execute)
    3. On failure: tool.handle_failure(request, error)
    4. On retry: tool.retry(request, attempt)

    Constraints
    -----------
    - Tools MUST be stateless between calls.
    - Tools MUST NOT call other tools or agents directly.
    - All state must travel through ToolRequest/ToolResponse.
    - Tools MUST be independently unit-testable.

    Subclassing
    -----------
    Implement these four abstract methods:
        execute()        — core async logic
        validate_input() — pre-execution validation
        handle_failure() — error normalisation
        retry()          — retry strategy hook
    """

    # Concrete tools set this to their registered pipeline name.
    tool_name: str = "base_tool"

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    async def execute(self, request: "ToolRequest") -> "ToolResponse":
        """
        Core tool execution logic.

        Parameters
        ----------
        request : ToolRequest
            Validated tool invocation request.

        Returns
        -------
        ToolResponse
            The tool's result. stdout/stderr must always be set.

        Raises
        ------
        ToolExecutionError
            On unrecoverable or retryable execution failures.

        Constraints
        -----------
        - Do NOT call other tools or agents.
        - Do NOT mutate the request object.
        - Set stdout/stderr regardless of success/failure.
        - Set latency_ms as accurately as possible.
        """
        ...

    @abstractmethod
    def validate_input(self, request: "ToolRequest") -> None:
        """
        Validate a ToolRequest before execution begins.

        Parameters
        ----------
        request : ToolRequest
            The request to validate.

        Raises
        ------
        ToolValidationError
            If the request is missing required parameters or contains
            invalid values for this tool.

        Notes
        -----
        Called by safe_execute() BEFORE execute(). Check required keys in
        request.parameters, type constraints, value ranges, etc.
        """
        ...

    @abstractmethod
    def handle_failure(
        self,
        request: "ToolRequest",
        error: Exception,
        attempt: int,
    ) -> "ToolFailure":
        """
        Normalise an exception into a structured ToolFailure.

        Parameters
        ----------
        request : ToolRequest
            The originating request.
        error : Exception
            The exception that occurred.
        attempt : int
            Which attempt number (1-indexed) this failure came from.

        Returns
        -------
        ToolFailure
            A fully populated failure record.

        Notes
        -----
        Called by safe_execute() after all retries are exhausted.
        Concrete tools should categorise the error type appropriately.
        """
        ...

    @abstractmethod
    async def retry(
        self,
        request: "ToolRequest",
        attempt: int,
    ) -> "ToolResponse":
        """
        Execute a single retry attempt.

        Parameters
        ----------
        request : ToolRequest
            The original request to retry.
        attempt : int
            Which retry attempt this is (1-indexed; attempt=1 is the first retry).

        Returns
        -------
        ToolResponse
            Result of this retry attempt.

        Notes
        -----
        In most cases this can simply call self.execute(request).
        Override to modify the request between attempts (e.g. back-off,
        token refresh, alternate endpoints).
        """
        ...

    # ------------------------------------------------------------------
    # Concrete orchestration helpers
    # ------------------------------------------------------------------

    async def safe_execute(
        self, request: "ToolRequest"
    ) -> tuple["ToolResponse", "ToolFailure | None"]:
        """
        Orchestrator-facing wrapper with full retry loop.

        Calls validate_input → execute → (on failure) retry up to
        retry_metadata.max_retries times → handle_failure on exhaustion.

        Parameters
        ----------
        request : ToolRequest
            The tool invocation request.

        Returns
        -------
        tuple[ToolResponse, ToolFailure | None]
            On success: (ToolResponse(status=SUCCESS), None)
            On exhausted retries: (ToolResponse(status=FAILURE), ToolFailure)

        Notes
        -----
        - Do NOT override this in concrete tools.
        - All timing and retry state is managed here.
        """
        try:
            self.validate_input(request)
        except ToolValidationError as e:
            failure = self.handle_failure(request, e, attempt=0)
            response = ToolResponse(
                tool_name=self.tool_name,
                correlation_id=request.correlation_id,
                job_id=request.job_id,
                status=ToolStatus.FAILURE,
                stdout="",
                stderr=str(e),
                latency_ms=0.0,
                error_message=str(e),
            )
            return response, failure

        last_error: Exception | None = None
        attempt = 0
        started_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()

        while attempt <= request.retry_metadata.max_retries:
            try:
                if attempt == 0:
                    response = await self.execute(request)
                else:
                    response = await self.retry(request, attempt)

                latency_ms = (time.perf_counter() - t0) * 1000.0
                self._logger.info(
                    "tool=%s attempt=%d status=success latency_ms=%.1f",
                    self.tool_name,
                    attempt,
                    latency_ms,
                )
                return response, None

            except (ToolExecutionError, Exception) as e:
                last_error = e
                is_retryable = getattr(e, "is_retryable", True)

                self._logger.warning(
                    "tool=%s attempt=%d error=%s retryable=%s",
                    self.tool_name,
                    attempt,
                    str(e),
                    is_retryable,
                )

                if not is_retryable:
                    break

                if attempt < request.retry_metadata.max_retries:
                    delay_ms = request.retry_metadata.next_delay_ms()
                    await asyncio.sleep(delay_ms / 1000.0)

                attempt += 1

        # All retries exhausted
        latency_ms = (time.perf_counter() - t0) * 1000.0
        failure = self.handle_failure(
            request, last_error or Exception("unknown"), attempt
        )
        response = ToolResponse(
            tool_name=self.tool_name,
            correlation_id=request.correlation_id,
            job_id=request.job_id,
            status=ToolStatus.FAILURE,
            stdout="",
            stderr=str(last_error),
            latency_ms=latency_ms,
            retries_used=attempt,
            started_at=started_at,
            error_message=str(last_error),
        )
        return response, failure

    def tool_info(self) -> dict[str, Any]:
        """Return a metadata dict describing this tool."""
        return {
            "tool_name": self.tool_name,
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
        }


# ---------------------------------------------------------------------------
# Standalone debug entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from contracts.tool_contracts import (
        RetryMetadata,
        RetryStrategy,
        ToolFailure,
        ToolRequest,
        ToolResponse,
        ToolStatus,
    )

    print("=" * 60)
    print("base_tool.py — standalone debug mode")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Concrete tool that succeeds on the 2nd attempt
    # -----------------------------------------------------------------------

    class FlakySearchTool(BaseTool):
        """Tool that fails on attempt 0, succeeds on attempt 1."""

        tool_name = "flaky_search"
        _attempt_count = 0

        def validate_input(self, request: ToolRequest) -> None:
            if "query" not in request.parameters:
                raise ToolValidationError(
                    self.tool_name, "Missing required parameter: 'query'."
                )

        async def execute(self, request: ToolRequest) -> ToolResponse:
            self.__class__._attempt_count += 1
            if self.__class__._attempt_count < 2:
                raise ToolExecutionError(
                    self.tool_name,
                    "Simulated transient failure.",
                    is_retryable=True,
                )
            return ToolResponse(
                tool_name=self.tool_name,
                correlation_id=request.correlation_id,
                job_id=request.job_id,
                status=ToolStatus.SUCCESS,
                stdout='{"result": "Paris"}',
                stderr="",
                result={"result": "Paris"},
                latency_ms=120.0,
            )

        def handle_failure(
            self, request: ToolRequest, error: Exception, attempt: int
        ) -> ToolFailure:
            return ToolFailure(
                tool_name=self.tool_name,
                correlation_id=request.correlation_id,
                job_id=request.job_id,
                error_type="transient",
                error_message=str(error),
                traceback=traceback.format_exc(),
                retry_metadata=RetryMetadata(
                    max_retries=request.retry_metadata.max_retries,
                    retries_used=attempt,
                ),
                is_retryable=True,
            )

        async def retry(self, request: ToolRequest, attempt: int) -> ToolResponse:
            return await self.execute(request)

    # -----------------------------------------------------------------------
    # Run safe_execute
    # -----------------------------------------------------------------------

    req = ToolRequest(
        tool_name="flaky_search",
        job_id="job-debug",
        agent_name="retrieval_agent",
        parameters={"query": "capital of France"},
        retry_metadata=RetryMetadata(
            max_retries=3,
            retry_strategy=RetryStrategy.FIXED,
            base_delay_ms=10.0,  # fast for debug
        ),
    )

    tool = FlakySearchTool()
    print("\n[tool_info]")
    import json
    print(json.dumps(tool.tool_info(), indent=2))

    response, failure = asyncio.run(tool.safe_execute(req))
    print(f"\n[safe_execute] success={response.success} retries_used={response.retries_used}")
    print(f"  stdout={response.stdout}")
    print(f"  failure={failure}")

    # -----------------------------------------------------------------------
    # Validation failure (missing 'query' param)
    # -----------------------------------------------------------------------

    bad_req = ToolRequest(
        tool_name="flaky_search",
        job_id="job-debug",
        agent_name="retrieval_agent",
        parameters={},  # missing 'query'
    )
    FlakySearchTool._attempt_count = 0
    resp2, fail2 = asyncio.run(tool.safe_execute(bad_req))
    print(f"\n[validation failure] success={resp2.success} error={resp2.error_message}")

    # -----------------------------------------------------------------------
    # Abstract instantiation guard
    # -----------------------------------------------------------------------
    print("\n[Instantiating BaseTool directly should raise]")
    try:
        BaseTool()  # type: ignore[abstract]
    except TypeError as e:
        print(f"  Caught expected error: {type(e).__name__}")

    print("\n✅ base_tool.py debug complete.")