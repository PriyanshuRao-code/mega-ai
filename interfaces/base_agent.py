# interfaces/base_agent.py (GENERATED ALONG WITH agents)

"""
interfaces/base_agent.py
========================
Abstract base interface for all agents in the multi-agent LLM system.

SOLID Alignment:
  - (I) Interface Segregation: minimal, focused interface
  - (D) Dependency Inversion: all agents depend on this abstraction

Imports:
  - abc.ABC, abc.abstractmethod  (stdlib)
  - contracts.shared_context.SharedContext
  - contracts.agent_contracts.*Result  (resolved at runtime via generics)

Exceptions:
  - AgentExecutionError  : raised when agent processing fails unrecoverably
  - AgentValidationError : raised when input/output schema validation fails
  - AgentTimeoutError    : raised when execution exceeds allowed duration
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent-level exceptions
# ---------------------------------------------------------------------------

class AgentError(Exception):
    """Base class for all agent errors."""
    def __init__(self, agent_name: str, message: str) -> None:
        self.agent_name = agent_name
        super().__init__(f"[{agent_name}] {message}")


class AgentExecutionError(AgentError):
    """Raised when an agent fails during core processing."""


class AgentValidationError(AgentError):
    """Raised when input or output fails schema validation."""


class AgentTimeoutError(AgentError):
    """Raised when agent execution exceeds the configured time limit."""


# ---------------------------------------------------------------------------
# Generic type variables
# ---------------------------------------------------------------------------

TInput = TypeVar("TInput")   # Always SharedContext in this system
TOutput = TypeVar("TOutput") # One of the *Result dataclasses


# ---------------------------------------------------------------------------
# Base agent interface
# ---------------------------------------------------------------------------

class BaseAgent(ABC, Generic[TInput, TOutput]):
    """
    Abstract base for every agent in the pipeline.

    Subclasses MUST implement:
        run(context: TInput) -> TOutput

    Subclasses MAY override:
        validate_input(context)   — pre-run schema guard
        validate_output(result)   — post-run schema guard
        agent_name (property)

    Lifecycle enforced by __call__:
        validate_input → run → validate_output → return result
    """

    # Maximum seconds a single run() is allowed to take (override per agent)
    TIMEOUT_SECONDS: float = 60.0

    # ------------------------------------------------------------------ #
    #  Identity                                                            #
    # ------------------------------------------------------------------ #

    @property
    def agent_name(self) -> str:
        return self.__class__.__name__

    # ------------------------------------------------------------------ #
    #  Core contract                                                       #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def run(self, context: TInput) -> TOutput:
        """
        Execute the agent's primary logic.

        Args:
            context: validated TInput (SharedContext instance)

        Returns:
            TOutput result dataclass

        Raises:
            AgentExecutionError: on unrecoverable processing failure
        """

    # ------------------------------------------------------------------ #
    #  Validation hooks (override to add schema checks)                   #
    # ------------------------------------------------------------------ #

    def validate_input(self, context: TInput) -> None:
        """
        Validate agent input before run().

        Args:
            context: the incoming context object

        Raises:
            AgentValidationError: if validation fails
        """
        if context is None:
            raise AgentValidationError(self.agent_name, "Input context must not be None")

    def validate_output(self, result: TOutput) -> None:
        """
        Validate agent output after run().

        Args:
            result: the result produced by run()

        Raises:
            AgentValidationError: if validation fails
        """
        if result is None:
            raise AgentValidationError(self.agent_name, "Output result must not be None")

    # ------------------------------------------------------------------ #
    #  Orchestrated entry-point (final — not overridable)                 #
    # ------------------------------------------------------------------ #

    @final
    def __call__(self, context: TInput) -> TOutput:
        """
        Orchestrated entry-point: validate → run (with timeout guard) → validate.

        Args:
            context: TInput

        Returns:
            TOutput

        Raises:
            AgentValidationError : input or output schema violation
            AgentExecutionError  : run() raised an unexpected error
            AgentTimeoutError    : run() exceeded TIMEOUT_SECONDS
        """
        log = logging.getLogger(self.agent_name)
        log.info("Starting | agent=%s", self.agent_name)

        # --- input validation ---
        try:
            self.validate_input(context)
        except AgentValidationError:
            raise
        except Exception as exc:
            raise AgentValidationError(self.agent_name, f"Unexpected validation error: {exc}") from exc

        # --- timed execution ---
        start = time.monotonic()
        try:
            result = self.run(context)
        except (AgentExecutionError, AgentValidationError, AgentTimeoutError):
            raise
        except Exception as exc:
            raise AgentExecutionError(self.agent_name, f"Unhandled error during run: {exc}") from exc
        finally:
            elapsed = time.monotonic() - start
            log.info("Finished | agent=%s elapsed=%.3fs", self.agent_name, elapsed)
            if elapsed > self.TIMEOUT_SECONDS:
                raise AgentTimeoutError(
                    self.agent_name,
                    f"Execution took {elapsed:.1f}s, limit is {self.TIMEOUT_SECONDS}s"
                )

        # --- output validation ---
        try:
            self.validate_output(result)
        except AgentValidationError:
            raise
        except Exception as exc:
            raise AgentValidationError(self.agent_name, f"Unexpected output validation error: {exc}") from exc

        return result
