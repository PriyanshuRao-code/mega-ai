# interfaces/base_tool.py (GENERATED ALONG WITH tools)
"""
interfaces/base_tool.py
=======================
Abstract base interface for all tools in the multi-agent system.

Imports   : abc, typing
Inputs    : ToolRequest (from contracts.tool_contracts)
Outputs   : ToolResponse (from contracts.tool_contracts)
Exceptions: NotImplementedError (if subclass skips execute/validate)
Dependencies: contracts.tool_contracts, contracts.shared_context
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import final

from contracts.tool_contracts import ToolRequest, ToolResponse, ToolStatus
from contracts.shared_context import SharedContext

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for every tool in the tooling layer.

    Subclasses MUST implement:
        - execute(request, context) -> ToolResponse
        - validate(request)        -> None  (raise on invalid)

    Subclasses SHOULD NOT override:
        - run()   — the template-method entry point (marked @final)
    """

    # ------------------------------------------------------------------ #
    #  Identity — subclasses set these as class-level constants            #
    # ------------------------------------------------------------------ #
    TOOL_NAME: str = "base_tool"
    VERSION: str = "1.0.0"
    MAX_RETRIES: int = 3
    TIMEOUT_SECONDS: float = 30.0

    # ------------------------------------------------------------------ #
    #  Template method (final — do NOT override)                           #
    # ------------------------------------------------------------------ #
    @final
    def run(self, request: ToolRequest, context: SharedContext) -> ToolResponse:
        """
        Entry point for all tool invocations.

        Enforces:
          1. Input validation
          2. Retry loop with exponential back-off
          3. Uniform error wrapping into ToolResponse
        """
        import time

        logger.info("[%s] run() called | request_id=%s", self.TOOL_NAME, request.request_id)

        last_exc: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self.validate(request)
                response = self.execute(request, context)
                response.attempts = attempt
                logger.info(
                    "[%s] success on attempt %d | request_id=%s",
                    self.TOOL_NAME, attempt, request.request_id,
                )
                return response

            except (ValueError, TypeError) as exc:
                # Malformed input — do NOT retry; surface immediately
                logger.warning("[%s] malformed input: %s", self.TOOL_NAME, exc)
                return ToolResponse.failure(
                    request_id=request.request_id,
                    tool_name=self.TOOL_NAME,
                    status=ToolStatus.INVALID_INPUT,
                    error=str(exc),
                    attempts=attempt,
                )

            except TimeoutError as exc:
                last_exc = exc
                logger.warning(
                    "[%s] timeout on attempt %d/%d", self.TOOL_NAME, attempt, self.MAX_RETRIES
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(2 ** (attempt - 1))   # 1 s, 2 s, 4 s …

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "[%s] error on attempt %d/%d: %s",
                    self.TOOL_NAME, attempt, self.MAX_RETRIES, exc,
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(2 ** (attempt - 1))

        return ToolResponse.failure(
            request_id=request.request_id,
            tool_name=self.TOOL_NAME,
            status=ToolStatus.ERROR,
            error=str(last_exc),
            attempts=self.MAX_RETRIES,
        )

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                  #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def execute(self, request: ToolRequest, context: SharedContext) -> ToolResponse:
        """
        Core tool logic.  Must return a ToolResponse — never raise for
        expected error conditions (use ToolResponse.failure() instead).
        Unexpected exceptions bubble up to run() which handles retries.
        """

    @abstractmethod
    def validate(self, request: ToolRequest) -> None:
        """
        Validate the incoming request.
        Raise ValueError  for missing / wrong-type fields.
        Raise TypeError   for structural mismatches.
        """
