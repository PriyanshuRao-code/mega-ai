"""
api/services.py
===============
Purpose     : Abstract service interfaces (ports) consumed by route handlers.
              Concrete adapters live outside this package.
Imports     : abc, typing, models
Outputs     : Abstract base classes only — no I/O
Dependencies: api.models
Exceptions  : NotImplementedError from un-implemented abstract methods;
              concrete implementations may raise RunNotFoundError,
              EvalNotFoundError, RewriteConflictError, ServiceUnavailableError
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from api.models import (
    EvalSummaryResponse,
    ExecutionTraceResponse,
    QueryRequest,
    QueryResponse,
    ReEvalRequest,
    ReEvalResponse,
    RewriteDecisionRequest,
    RewriteDecisionResponse,
    SSEEvent,
)


# ---------------------------------------------------------------------------
# Domain exceptions  (raised by concrete service implementations)
# ---------------------------------------------------------------------------

class RunNotFoundError(Exception):
    """Raised when a run_id cannot be resolved."""


class EvalNotFoundError(Exception):
    """Raised when no eval exists for the given run_id."""


class RewriteConflictError(Exception):
    """Raised when a rewrite decision has already been recorded."""


class ServiceUnavailableError(Exception):
    """Raised when a downstream dependency is temporarily unavailable."""


# ---------------------------------------------------------------------------
# Port interfaces
# ---------------------------------------------------------------------------

class IQueryService(ABC):
    @abstractmethod
    async def submit(self, request: QueryRequest) -> QueryResponse:
        """
        Input : QueryRequest
        Output: QueryResponse
        Raises: ServiceUnavailableError
        """

    @abstractmethod
    async def stream(self, request: QueryRequest, run_id: str | None = None) -> AsyncIterator[SSEEvent]:
        """
        Input : QueryRequest
        Output: async generator of SSEEvent
        Raises: ServiceUnavailableError
        """


class ITraceService(ABC):
    @abstractmethod
    async def get_trace(self, run_id: str) -> ExecutionTraceResponse:
        """
        Input : run_id (str)
        Output: ExecutionTraceResponse
        Raises: RunNotFoundError
        """


class IEvalService(ABC):
    @abstractmethod
    async def get_latest_eval(self, run_id: str) -> EvalSummaryResponse:
        """
        Input : run_id (str)
        Output: EvalSummaryResponse
        Raises: RunNotFoundError, EvalNotFoundError
        """

    @abstractmethod
    async def reeval(self, run_id: str, request: ReEvalRequest) -> ReEvalResponse:
        """
        Input : run_id (str), ReEvalRequest
        Output: ReEvalResponse
        Raises: RunNotFoundError
        """


class IRewriteService(ABC):
    @abstractmethod
    async def decide(
        self, run_id: str, request: RewriteDecisionRequest
    ) -> RewriteDecisionResponse:
        """
        Input : run_id (str), RewriteDecisionRequest
        Output: RewriteDecisionResponse
        Raises: RunNotFoundError, RewriteConflictError
        """
