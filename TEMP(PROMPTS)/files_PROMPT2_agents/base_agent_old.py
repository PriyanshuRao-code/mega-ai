# interfaces/base_agent.py (ORIGINAL)
"""
Module: interfaces/base_agent.py
==================================
Purpose:
    Defines the abstract BaseAgent interface that every agent in the
    multi-agent pipeline must implement. Enforces the contract:
    agents receive SharedContext, produce BaseAgentOutput, and may
    NOT call other agents directly.

    All communication between agents happens exclusively through
    SharedContext mutation and the pipeline orchestrator.

Input Datatypes:
    SharedContext — from contracts/shared_context.py

Output Datatypes:
    BaseAgentOutput — from contracts/agent_contracts.py

Dependencies:
    - contracts/shared_context.py
    - contracts/agent_contracts.py
    - Python 3.11+
    - asyncio (stdlib)
    - abc (stdlib)
    - logging (stdlib)

SOLID Principle:
    Open/Closed — new agents extend BaseAgent without modifying it.
    Dependency Inversion — orchestrator depends on BaseAgent abstraction,
    not concrete agent classes.
    Interface Segregation — BaseAgent only declares the agent lifecycle;
    tool access is declared in BaseTool.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

# NOTE: These imports will resolve once the project package is on PYTHONPATH.
# For standalone debug mode they are imported with a try/except fallback.
try:
    from contracts.agent_contracts import BaseAgentInput, BaseAgentOutput, AgentStatus
    from contracts.shared_context import SharedContext
except ImportError:
    # Allow the file to be syntax-checked in isolation
    BaseAgentInput = None  # type: ignore[assignment,misc]
    BaseAgentOutput = None  # type: ignore[assignment,misc]
    AgentStatus = None  # type: ignore[assignment,misc]
    SharedContext = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context budget descriptor
# ---------------------------------------------------------------------------


class ContextBudget:
    """
    Describes the token budget an agent requests from the pipeline.

    Fields
    ------
    max_input_tokens : int
        Maximum tokens the agent will consume from SharedContext.
    max_output_tokens : int
        Maximum tokens the agent is allowed to generate.
    reserved_system_tokens : int
        Tokens reserved for system prompt overhead.
    """

    def __init__(
        self,
        max_input_tokens: int = 4_096,
        max_output_tokens: int = 1_024,
        reserved_system_tokens: int = 512,
    ) -> None:
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.reserved_system_tokens = reserved_system_tokens

    @property
    def total_budget(self) -> int:
        return (
            self.max_input_tokens
            + self.max_output_tokens
            + self.reserved_system_tokens
        )

    def __repr__(self) -> str:
        return (
            f"ContextBudget("
            f"input={self.max_input_tokens}, "
            f"output={self.max_output_tokens}, "
            f"system={self.reserved_system_tokens}, "
            f"total={self.total_budget})"
        )


# ---------------------------------------------------------------------------
# Abstract BaseAgent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """
    Abstract base class that all pipeline agents must subclass.

    Lifecycle
    ---------
    1. Orchestrator calls agent.validate_input(context)
    2. Orchestrator calls output = await agent.execute(context)
    3. Orchestrator calls agent.validate_output(output)
    4. Orchestrator calls agent.log_execution(context, output, latency_ms)

    Constraints (enforced by convention; see execute() docstring)
    ------------------------------------------------------------
    - Agents MUST NOT call other agents directly.
    - Agents MUST NOT import or instantiate other agent classes.
    - All inter-agent data sharing goes through SharedContext.
    - Agents MUST be independently unit-testable.

    Subclassing
    -----------
    Implement the four abstract methods:
        execute()         — core async logic
        validate_input()  — pre-execution guard
        validate_output() — post-execution guard
        get_context_budget() — declare token requirements
    """

    # Subclasses set this to their registered pipeline name.
    agent_name: str = "base_agent"

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    # ------------------------------------------------------------------
    # Abstract methods (MUST be implemented by every concrete agent)
    # ------------------------------------------------------------------

    @abstractmethod
    async def execute(self, context: "SharedContext") -> "BaseAgentOutput":
        """
        Core agent logic.

        Parameters
        ----------
        context : SharedContext
            The shared pipeline state. Read agent_outputs from previous
            agents here. Write your output via context.record_agent_output().

        Returns
        -------
        BaseAgentOutput (or a subclass)
            The agent's result. Must also be persisted into context via
            context.record_agent_output(self.agent_name, output.model_dump()).

        Raises
        ------
        AgentExecutionError
            On unrecoverable failures. The orchestrator catches this and
            marks the context with a policy violation or failure trace.

        Constraints
        -----------
        - Do NOT instantiate or call other BaseAgent subclasses.
        - Do NOT access databases, APIs, or tools directly — use BaseTool.
        - Do NOT mutate SharedContext fields other than agent_outputs,
          citations, token_usage, and execution_trace.
        """
        ...

    @abstractmethod
    def validate_input(self, context: "SharedContext") -> None:
        """
        Validate that SharedContext contains everything this agent needs
        before execute() is called.

        Parameters
        ----------
        context : SharedContext
            The current pipeline state.

        Raises
        ------
        ValueError
            If required fields are missing or invalid.

        Notes
        -----
        Called by the orchestrator BEFORE execute(). Agents should check
        for required keys in context.agent_outputs, non-empty query, etc.
        """
        ...

    @abstractmethod
    def validate_output(self, output: "BaseAgentOutput") -> None:
        """
        Validate the output produced by this agent after execute() returns.

        Parameters
        ----------
        output : BaseAgentOutput
            The output to validate.

        Raises
        ------
        ValueError
            If the output fails post-conditions.

        Notes
        -----
        Called by the orchestrator AFTER execute(). Check confidence
        thresholds, required fields, schema correctness, etc.
        """
        ...

    @abstractmethod
    def get_context_budget(self) -> ContextBudget:
        """
        Declare the token budget this agent requires.

        Returns
        -------
        ContextBudget
            The agent's token budget declaration.

        Notes
        -----
        The orchestrator uses this to enforce context window limits and
        decide whether to run compression agents before this agent.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete lifecycle helpers (may be overridden, but not required)
    # ------------------------------------------------------------------

    def log_execution(
        self,
        context: "SharedContext",
        output: "BaseAgentOutput",
        latency_ms: float,
    ) -> None:
        """
        Log execution details to the shared context trace and Python logger.

        Parameters
        ----------
        context : SharedContext
            The shared pipeline state (mutated: execution_trace appended).
        output : BaseAgentOutput
            The agent's output (used for status/confidence logging).
        latency_ms : float
            Wall-clock execution time in milliseconds.

        Notes
        -----
        Concrete agents may override this to add custom log fields, but
        should always call super().log_execution() first.
        """
        status = getattr(output, "status", "unknown")
        confidence = getattr(output, "confidence_score", None)

        self._logger.info(
            "agent=%s status=%s confidence=%s latency_ms=%.1f job_id=%s",
            self.agent_name,
            status,
            confidence,
            latency_ms,
            context.job_id,
        )

        context.trace(
            agent_name=self.agent_name,
            step="completed",
            duration_ms=latency_ms,
            notes=f"status={status} confidence={confidence}",
        )

    async def safe_execute(
        self, context: "SharedContext"
    ) -> "BaseAgentOutput":
        """
        Orchestrator-facing wrapper that times execute() and calls
        log_execution() automatically.

        Parameters
        ----------
        context : SharedContext
            The shared pipeline state.

        Returns
        -------
        BaseAgentOutput
            The agent's validated output.

        Notes
        -----
        This method is called by the orchestrator, not by agent code.
        Do NOT override this in concrete agents — override execute() instead.
        """
        context.set_current_agent(self.agent_name)
        context.trace(self.agent_name, "started")

        self.validate_input(context)

        t0 = time.perf_counter()
        output = await self.execute(context)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self.validate_output(output)
        self.log_execution(context, output, latency_ms)

        return output

    def agent_info(self) -> dict[str, Any]:
        """Return a metadata dict describing this agent."""
        budget = self.get_context_budget()
        return {
            "agent_name": self.agent_name,
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "context_budget": {
                "max_input_tokens": budget.max_input_tokens,
                "max_output_tokens": budget.max_output_tokens,
                "reserved_system_tokens": budget.reserved_system_tokens,
                "total_budget": budget.total_budget,
            },
        }


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class AgentExecutionError(Exception):
    """
    Raised by a BaseAgent subclass when it encounters an unrecoverable error.

    Attributes
    ----------
    agent_name : str
        Name of the agent that raised the error.
    job_id : str
        Pipeline run ID for tracing.
    original_exception : Exception | None
        The underlying exception that caused this failure.
    """

    def __init__(
        self,
        message: str,
        agent_name: str = "unknown",
        job_id: str = "unknown",
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.agent_name = agent_name
        self.job_id = job_id
        self.original_exception = original_exception

    def __repr__(self) -> str:
        return (
            f"AgentExecutionError(agent={self.agent_name!r}, "
            f"job_id={self.job_id!r}, msg={str(self)!r})"
        )


# ---------------------------------------------------------------------------
# Standalone debug entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import sys
    import os

    # Make the project root importable
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from contracts.agent_contracts import (
        AgentStatus,
        BaseAgentOutput,
        ProvenanceMetadata,
    )
    from contracts.shared_context import SharedContext

    print("=" * 60)
    print("base_agent.py — standalone debug mode")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Build a minimal concrete agent for testing
    # -----------------------------------------------------------------------

    class EchoAgent(BaseAgent):
        """Minimal concrete agent that echoes the query back as output."""

        agent_name = "echo_agent"

        async def execute(self, context: SharedContext) -> BaseAgentOutput:
            output = BaseAgentOutput(
                job_id=context.job_id,
                agent_name=self.agent_name,
                status=AgentStatus.SUCCESS,
                confidence_score=1.0,
                provenance=ProvenanceMetadata(agent_name=self.agent_name),
            )
            context.record_agent_output(self.agent_name, output.model_dump())
            return output

        def validate_input(self, context: SharedContext) -> None:
            if not context.query:
                raise ValueError("EchoAgent requires a non-empty query.")

        def validate_output(self, output: BaseAgentOutput) -> None:
            if output.status == AgentStatus.FAILURE:
                raise ValueError("EchoAgent output must not be FAILURE.")

        def get_context_budget(self) -> ContextBudget:
            return ContextBudget(
                max_input_tokens=512,
                max_output_tokens=256,
                reserved_system_tokens=128,
            )

    # -----------------------------------------------------------------------
    # Run the agent
    # -----------------------------------------------------------------------

    ctx = SharedContext(query="What is the capital of France?")
    agent = EchoAgent()

    print("\n[agent_info]")
    import json
    print(json.dumps(agent.agent_info(), indent=2))

    print("\n[safe_execute]")
    result = asyncio.run(agent.safe_execute(ctx))
    print(f"  status={result.status}")
    print(f"  confidence={result.confidence_score}")
    print(f"  agents_completed={list(ctx.agent_outputs.keys())}")
    print(f"  trace_steps={len(ctx.execution_trace)}")

    # -----------------------------------------------------------------------
    # Test AgentExecutionError
    # -----------------------------------------------------------------------
    print("\n[AgentExecutionError repr]")
    err = AgentExecutionError(
        message="Retrieval timed out.",
        agent_name="retrieval_agent",
        job_id=ctx.job_id,
        original_exception=TimeoutError("upstream timeout"),
    )
    print(f"  {err!r}")

    # -----------------------------------------------------------------------
    # Validate that unimplemented abstract methods raise TypeError
    # -----------------------------------------------------------------------
    print("\n[Instantiating BaseAgent directly should raise]")
    try:
        BaseAgent()  # type: ignore[abstract]
    except TypeError as e:
        print(f"  Caught expected error: {type(e).__name__}")

    print("\n✅ base_agent.py debug complete.")