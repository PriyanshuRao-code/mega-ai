# interfaces/base.py (along with orchestrator)
"""
interfaces/base.py
────────────────────────────────────────────────────────────────────────────────
Abstract interface contracts for the multi-agent system.

DO NOT MODIFY — implemented by agents and orchestrator components.

Imports  : abc, typing, contracts.models
Exports  : BaseAgent, IRouter, IScheduler, IRetryManager, IOrchestrator
Exceptions: NotImplementedError (raised implicitly for unimplemented abstract methods)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from contracts.models import (
    AgentExecutionEvent,
    ExecutionEvent,
    SharedContext,
    ToolResponse,
)


# ──────────────────────────────────────────────────────────────────────────────
# Agent interface
# ──────────────────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Contract every concrete agent must fulfil.

    Responsibilities
    ----------------
    - Declare a unique name.
    - Declare which token budget it expects (max_tokens).
    - Execute against a SharedContext and return a ToolResponse.
    - Optionally validate that execution is safe before proceeding.

    Input  : SharedContext
    Output : ToolResponse
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Globally unique agent identifier."""
        ...

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Maximum tokens this agent may consume in a single call."""
        ...

    @abstractmethod
    async def execute(self, context: SharedContext) -> ToolResponse:
        """
        Run the agent against the current shared context.

        Parameters
        ----------
        context : SharedContext
            Current shared state — read freely, do NOT mutate directly.

        Returns
        -------
        ToolResponse
            Structured result including success flag, output, and token usage.

        Raises
        ------
        asyncio.TimeoutError  : If the agent exceeds its time budget.
        RuntimeError          : For unrecoverable internal errors.
        """
        ...

    async def validate(self, context: SharedContext) -> bool:
        """
        Optional pre-flight check.  Default: always valid.

        Returns True when the agent is safe to execute; False to skip.
        """
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Router interface
# ──────────────────────────────────────────────────────────────────────────────

class IRouter(ABC):
    """
    Contract for the dynamic routing component.

    Responsibilities
    ----------------
    - Inspect SharedContext and select the best next agent to run.
    - Justify and log every routing decision.
    - Signal ambiguity when the decision cannot be made confidently.

    Input  : SharedContext
    Output : str (agent name) or None when routing is complete
    """

    @abstractmethod
    async def select_next_agent(self, context: SharedContext) -> Optional[str]:
        """
        Choose the next agent to execute.

        Parameters
        ----------
        context : SharedContext

        Returns
        -------
        str | None
            Agent name, or None if orchestration should stop.

        Raises
        ------
        ValueError : When no valid routing decision can be made.
        """
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler interface
# ──────────────────────────────────────────────────────────────────────────────

class IScheduler(ABC):
    """
    Contract for the dependency-graph-aware scheduler.

    Responsibilities
    ----------------
    - Build and validate the dependency graph stored in SharedContext.
    - Return which agents are ready to run (all dependencies satisfied).
    - Prevent premature execution of dependent agents.

    Input  : SharedContext
    Output : List[str] of ready agent names
    """

    @abstractmethod
    def get_ready_agents(self, context: SharedContext) -> List[str]:
        """
        Return agents whose dependencies have all completed successfully.

        Parameters
        ----------
        context : SharedContext

        Returns
        -------
        List[str]
            Agent names that can be scheduled immediately.

        Raises
        ------
        ValueError : If a dependency cycle is detected.
        """
        ...

    @abstractmethod
    def is_complete(self, context: SharedContext) -> bool:
        """Return True when all agents in the graph have finished."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Retry manager interface
# ──────────────────────────────────────────────────────────────────────────────

class IRetryManager(ABC):
    """
    Contract for retry / backoff logic.

    Responsibilities
    ----------------
    - Wrap agent execution with configurable retry attempts.
    - Apply exponential back-off between attempts.
    - Emit AgentExecutionEvents for every retry attempt.

    Input  : BaseAgent, SharedContext, event sink
    Output : ToolResponse (final outcome after all attempts)
    """

    @abstractmethod
    async def execute_with_retry(
        self,
        agent: "BaseAgent",
        context: SharedContext,
        event_sink: List[AgentExecutionEvent],
    ) -> ToolResponse:
        """
        Execute the agent, retrying on transient failures.

        Parameters
        ----------
        agent      : BaseAgent  — agent to run.
        context    : SharedContext — shared state.
        event_sink : list       — mutable list; append events here.

        Returns
        -------
        ToolResponse
            Last result (success or final failure).

        Raises
        ------
        asyncio.TimeoutError : If all attempts exhaust the time budget.
        RuntimeError         : For non-retryable errors.
        """
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator interface
# ──────────────────────────────────────────────────────────────────────────────

class IOrchestrator(ABC):
    """
    Top-level orchestration contract.

    Responsibilities
    ----------------
    - Accept a SharedContext, execute the full agent graph, return results.

    Input  : SharedContext
    Output : tuple[SharedContext, ExecutionEvent]
    """

    @abstractmethod
    async def run(
        self, context: SharedContext
    ) -> tuple[SharedContext, ExecutionEvent]:
        """
        Execute all agents as dictated by the dependency graph and router.

        Parameters
        ----------
        context : SharedContext

        Returns
        -------
        tuple[SharedContext, ExecutionEvent]
            Updated shared context + full structured log of the run.

        Raises
        ------
        RuntimeError : If orchestration fails unrecoverably.
        """
        ...
