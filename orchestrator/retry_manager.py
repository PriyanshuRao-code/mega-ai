"""
orchestrator/retry_manager.py
────────────────────────────────────────────────────────────────────────────────
Retry and timeout management for the multi-agent orchestration system.

Responsibility
--------------
Wrap any BaseAgent.execute() call with configurable retry attempts,
exponential back-off (with optional jitter), and per-attempt timeout
enforcement.  Emit structured AgentExecutionEvents for every attempt.

Imports
-------
    Standard : asyncio, logging, pydantic models, random, time, typing
    Internal : contracts.models, interfaces.base

Input datatype  : BaseAgent, SharedContext, List[AgentExecutionEvent]
Output datatype : ToolResponse  (final result — success or terminal failure)

Possible exceptions
-------------------
    asyncio.TimeoutError — propagated when all attempts exhaust the timeout.
    RuntimeError         — raised for non-retryable errors (policy violations,
                           budget exhaustion flagged by the agent).

Dependencies
------------
    contracts.models.{ToolResponse, AgentExecutionEvent, EventType, SharedContext}
    interfaces.base.{IRetryManager, BaseAgent}

SOLID notes
-----------
    S — retry / backoff logic only; does not route or schedule.
    O — RetryPolicy is a separate injectable dataclass; swap it to change
        back-off strategy without touching RetryManager.
    L — fulfils IRetryManager; ToolResponse semantics are preserved.
    I — depends only on BaseAgent and SharedContext abstractions.
    D — injects RetryPolicy; no internal construction of policy.
"""

from __future__ import annotations

import os as _os, sys as _sys
if __name__ == "__main__":
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import asyncio
import logging
import random
import time
from pydantic import BaseModel, Field
from typing import List, Optional, Set, Type

from contracts.models import (
    AgentExecutionEvent,
    EventType,
    ExecutionStatus,
    SharedContext,
    ToolResponse,
)
from interfaces.base_agent import BaseAgent
from orchestrator.interfaces import IRetryManager

logger = logging.getLogger("orchestrator.retry_manager")


# ──────────────────────────────────────────────────────────────────────────────
# Retry policy value object (Open for extension)
# ──────────────────────────────────────────────────────────────────────────────

class RetryPolicy(BaseModel):
    """
    Immutable configuration for retry behaviour.

    Fields
    ------
    max_attempts      : Total attempts allowed (first + retries).
    base_delay_s      : Initial back-off delay in seconds.
    max_delay_s       : Upper bound on exponential back-off.
    backoff_multiplier: Factor applied to delay on each retry.
    jitter            : Whether to add uniform random jitter (±20 % of delay).
    timeout_s         : Per-attempt time limit in seconds; None = no limit.
    non_retryable     : Exception types that should not trigger a retry.
    """
    max_attempts      : int              = 3
    base_delay_s      : float            = 1.0
    max_delay_s       : float            = 30.0
    backoff_multiplier: float            = 2.0
    jitter            : bool             = True
    timeout_s         : Optional[float]  = 30.0
    non_retryable     : tuple            = Field(
        default_factory=lambda: (ValueError, RuntimeError)
    )

    def delay_for_attempt(self, attempt: int) -> float:
        """
        Compute the sleep duration before the given retry attempt (1-indexed).

        Input  : int  (attempt index, 1 = first retry)
        Output : float (seconds to sleep)
        """
        raw   = self.base_delay_s * (self.backoff_multiplier ** (attempt - 1))
        delay = min(raw, self.max_delay_s)
        if self.jitter:
            delay *= random.uniform(0.8, 1.2)
        return delay


# ──────────────────────────────────────────────────────────────────────────────
# Retry manager implementation
# ──────────────────────────────────────────────────────────────────────────────

class ExponentialRetryManager(IRetryManager):
    """
    Concrete IRetryManager with exponential back-off and per-attempt timeout.

    Parameters
    ----------
    policy : RetryPolicy — configuration injected at construction time.
    """

    def __init__(self, policy: RetryPolicy = RetryPolicy()) -> None:
        self._policy = policy

    # ── IRetryManager contract ────────────────────────────────────────────────

    async def execute_with_retry(
        self,
        agent       : BaseAgent,
        context     : SharedContext,
        event_sink  : List[AgentExecutionEvent],
    ) -> ToolResponse:
        """
        Execute agent with retries, back-off, and timeout per attempt.

        Input
        -----
        agent      : BaseAgent — agent to execute.
        context    : SharedContext — shared state.
        event_sink : List[AgentExecutionEvent] — mutable; events appended here.

        Output
        ------
        ToolResponse — the final result (may be a failure after all retries).

        Raises
        ------
        RuntimeError         — for non-retryable exceptions after first attempt.
        asyncio.TimeoutError — if every attempt times out.
        """
        policy        = self._policy
        last_response : Optional[ToolResponse] = None
        last_exc      : Optional[Exception]    = None

        for attempt in range(1, policy.max_attempts + 1):
            self._emit(
                event_sink, context, EventType.AGENT_STARTED, agent.agent_name,
                {"attempt": attempt, "max_attempts": policy.max_attempts},
            )
            context.agent_statuses[agent.agent_name] = ExecutionStatus.RUNNING
            t0 = time.monotonic()

            try:
                last_response = await self._run_with_timeout(
                    agent, context, policy.timeout_s
                )
                elapsed = time.monotonic() - t0

                if last_response.success:
                    self._emit(
                        event_sink, context, EventType.AGENT_COMPLETED, agent.agent_name,
                        {
                            "attempt"     : attempt,
                            "elapsed_s"   : round(elapsed, 3),
                            "tokens_used" : last_response.tokens_used,
                        },
                    )
                    logger.info(
                        "Agent '%s' succeeded on attempt %d/%d (%.3fs).",
                        agent.agent_name, attempt, policy.max_attempts, elapsed,
                    )
                    return last_response

                # Agent returned success=False — treat as soft failure
                logger.warning(
                    "Agent '%s' returned failure on attempt %d/%d: %s",
                    agent.agent_name, attempt, policy.max_attempts, last_response.error,
                )
                self._emit(
                    event_sink, context, EventType.AGENT_FAILED, agent.agent_name,
                    {
                        "attempt"  : attempt,
                        "error"    : last_response.error,
                        "elapsed_s": round(elapsed, 3),
                    },
                )

            except asyncio.TimeoutError as exc:
                elapsed = time.monotonic() - t0
                logger.error(
                    "Agent '%s' TIMED OUT on attempt %d/%d after %.3fs.",
                    agent.agent_name, attempt, policy.max_attempts, elapsed,
                )
                self._emit(
                    event_sink, context, EventType.AGENT_TIMEOUT, agent.agent_name,
                    {"attempt": attempt, "timeout_s": policy.timeout_s},
                )
                last_exc      = exc
                last_response = ToolResponse(
                    agent_name=agent.agent_name,
                    output=None,
                    tokens_used=0,
                    success=False,
                    error=f"Timeout after {policy.timeout_s}s on attempt {attempt}.",
                )
                context.agent_statuses[agent.agent_name] = ExecutionStatus.TIMEOUT

            except tuple(policy.non_retryable) as exc:  # type: ignore[misc]
                # Non-retryable: fail immediately
                logger.error(
                    "Agent '%s' raised non-retryable %s: %s",
                    agent.agent_name, type(exc).__name__, exc,
                )
                self._emit(
                    event_sink, context, EventType.AGENT_FAILED, agent.agent_name,
                    {"attempt": attempt, "error": str(exc), "non_retryable": True},
                )
                context.agent_statuses[agent.agent_name] = ExecutionStatus.FAILED
                return ToolResponse(
                    agent_name=agent.agent_name,
                    output=None,
                    tokens_used=0,
                    success=False,
                    error=f"Non-retryable {type(exc).__name__}: {exc}",
                )

            except Exception as exc:  # noqa: BLE001
                elapsed = time.monotonic() - t0
                logger.warning(
                    "Agent '%s' raised %s on attempt %d/%d: %s",
                    agent.agent_name, type(exc).__name__, attempt, policy.max_attempts, exc,
                )
                self._emit(
                    event_sink, context, EventType.AGENT_FAILED, agent.agent_name,
                    {"attempt": attempt, "error": str(exc), "elapsed_s": round(elapsed, 3)},
                )
                last_exc = exc
                last_response = ToolResponse(
                    agent_name=agent.agent_name,
                    output=None,
                    tokens_used=0,
                    success=False,
                    error=str(exc),
                )

            # Back-off before next attempt (if not the last)
            if attempt < policy.max_attempts:
                delay = policy.delay_for_attempt(attempt)
                logger.debug(
                    "Back-off %.3fs before retry %d for agent '%s'.",
                    delay, attempt + 1, agent.agent_name,
                )
                self._emit(
                    event_sink, context, EventType.AGENT_RETRYING, agent.agent_name,
                    {"next_attempt": attempt + 1, "delay_s": round(delay, 3)},
                )
                await asyncio.sleep(delay)

        # All attempts exhausted
        context.agent_statuses[agent.agent_name] = ExecutionStatus.FAILED
        logger.error(
            "Agent '%s' failed permanently after %d attempt(s).",
            agent.agent_name, policy.max_attempts,
        )
        assert last_response is not None  # always set in loop
        return last_response

    # ── internal helpers ──────────────────────────────────────────────────────

    async def _run_with_timeout(
        self,
        agent    : BaseAgent,
        context  : SharedContext,
        timeout_s: Optional[float],
    ) -> ToolResponse:
        """
        Execute agent.execute() with an optional asyncio timeout.

        Input  : BaseAgent, SharedContext, Optional[float]
        Output : ToolResponse

        Raises
        ------
        asyncio.TimeoutError — if execution exceeds timeout_s.
        """
        return await asyncio.wait_for( asyncio.to_thread(agent, context), timeout=timeout_s, )

    @staticmethod
    def _emit(
        sink      : List[AgentExecutionEvent],
        context   : SharedContext,
        event_type: EventType,
        agent_name: str,
        metadata  : dict,
    ) -> None:
        """Append a structured event to the caller-supplied sink."""
        sink.append(
            AgentExecutionEvent(
                event_type=event_type,
                agent_name=agent_name,
                task_id=context.task_id,
                metadata=metadata,
            )
        )


# ──────────────────────────────────────────────────────────────────────────────
# Self-contained debug harness
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import logging
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from contracts.models import ExecutionStatus, SharedContext, ToolResponse
    from interfaces.base_agent import BaseAgent

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # ── Stub agents for testing ───────────────────────────────────────────────

    class AlwaysSucceedsAgent(BaseAgent):
        @property
        def name(self) -> str: return "always_ok"
        @property
        def max_tokens(self) -> int: return 500
        async def execute(self, context: SharedContext) -> ToolResponse:
            return ToolResponse(agent_name=self.name, output="ok",
                                tokens_used=10, success=True)

    class FailsTwiceThenSucceedsAgent(BaseAgent):
        def __init__(self): self._calls = 0
        @property
        def name(self) -> str: return "flaky_agent"
        @property
        def max_tokens(self) -> int: return 500
        async def execute(self, context: SharedContext) -> ToolResponse:
            self._calls += 1
            if self._calls < 3:
                return ToolResponse(agent_name=self.name, output=None,
                                    tokens_used=5, success=False,
                                    error=f"Transient failure #{self._calls}")
            return ToolResponse(agent_name=self.name, output="recovered",
                                tokens_used=20, success=True)

    class AlwaysTimesOutAgent(BaseAgent):
        @property
        def name(self) -> str: return "timeout_agent"
        @property
        def max_tokens(self) -> int: return 500
        async def execute(self, context: SharedContext) -> ToolResponse:
            await asyncio.sleep(99)   # will always be killed by timeout
            return ToolResponse(agent_name=self.name, output=None,
                                tokens_used=0, success=False, error="unreachable")

    class NonRetryableAgent(BaseAgent):
        @property
        def name(self) -> str: return "nr_agent"
        @property
        def max_tokens(self) -> int: return 500
        async def execute(self, context: SharedContext) -> ToolResponse:
            raise ValueError("Policy violation — stop immediately")

    async def _debug() -> None:
        print("\n" + "=" * 70)
        print("retry_manager.py — debug harness")
        print("=" * 70)

        fast_policy = RetryPolicy(
            max_attempts=3,
            base_delay_s=0.01,
            max_delay_s=0.05,
            timeout_s=0.5,
            jitter=False,
        )
        mgr = ExponentialRetryManager(policy=fast_policy)

        def _ctx(task_id: str, agent_name: str) -> SharedContext:
            return SharedContext(
                task_id=task_id,
                goal="debug",
                token_budget=50_000,
                available_agents=[agent_name],
                dependency_graph={agent_name: []},
                agent_statuses={agent_name: ExecutionStatus.PENDING},
            )

        events: list = []

        # Test 1: success on first attempt
        ctx = _ctx("t1", "always_ok")
        resp = await mgr.execute_with_retry(AlwaysSucceedsAgent(), ctx, events)
        print(f"\n[Test 1] Always-ok: success={resp.success}, output={resp.output}")
        assert resp.success

        # Test 2: flaky agent recovers on 3rd attempt
        events.clear()
        ctx = _ctx("t2", "flaky_agent")
        resp = await mgr.execute_with_retry(FailsTwiceThenSucceedsAgent(), ctx, events)
        print(f"[Test 2] Flaky agent: success={resp.success}, output={resp.output}")
        assert resp.success and resp.output == "recovered"

        # Test 3: timeout
        events.clear()
        ctx = _ctx("t3", "timeout_agent")
        resp = await mgr.execute_with_retry(AlwaysTimesOutAgent(), ctx, events)
        print(f"[Test 3] Timeout agent: success={resp.success}, error={resp.error}")
        assert not resp.success and "Timeout" in resp.error

        # Test 4: non-retryable
        events.clear()
        ctx = _ctx("t4", "nr_agent")
        resp = await mgr.execute_with_retry(NonRetryableAgent(), ctx, events)
        print(f"[Test 4] Non-retryable: success={resp.success}, error={resp.error}")
        assert not resp.success and "non_retryable" in str(events[-1].metadata)

        print(f"\nTotal events emitted across all tests: {len(events)}")
        print("\n✅  All retry_manager assertions passed.")

    asyncio.run(_debug())
