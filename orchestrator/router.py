"""
orchestrator/router.py
────────────────────────────────────────────────────────────────────────────────
Dynamic routing component for the multi-agent orchestration system.

Responsibility
--------------
Inspect SharedContext at runtime and select the best next agent to execute.
No chains are hardcoded — every decision is derived from context state,
dependency readiness (delegated to IScheduler), and goal-alignment scoring.

Imports
-------
    Standard : asyncio, logging, dataclasses, typing
    Internal : contracts.models, interfaces.base, orchestrator.scheduler

Input datatype  : SharedContext
Output datatype : Optional[str]  (agent name, or None = stop orchestration)

Possible exceptions
-------------------
    ValueError   — raised when routing is completely ambiguous and
                   no fallback heuristic resolves it.
    RuntimeError — raised if context is in an irrecoverable state
                   (budget exhausted, all agents failed).

Dependencies
------------
    IScheduler — used to query which agents are currently ready to execute.
    logging    — structured routing decisions emitted to "orchestrator.router".

SOLID notes
-----------
    S — single responsibility: routing logic only; no execution, no retry.
    O — extend by subclassing and overriding _score_candidate().
    L — fulfils IRouter without breaking its contract.
    I — depends only on IScheduler, not the full orchestrator.
    D — depends on abstractions (IScheduler, SharedContext), not concretions.
"""

from __future__ import annotations

import os as _os, sys as _sys
if __name__ == "__main__":
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from contracts.models import (
    AgentExecutionEvent,
    EventType,
    ExecutionStatus,
    SharedContext,
)
from orchestrator.interfaces import IRouter, IScheduler

logger = logging.getLogger("orchestrator.router")


# ──────────────────────────────────────────────────────────────────────────────
# Supporting value objects
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """
    Value object capturing the full rationale behind a routing choice.

    Fields
    ------
    selected_agent  : Agent chosen (None = stop).
    candidates      : All agents that were eligible at this step.
    scores          : Numeric score assigned to each candidate.
    reason          : Human-readable justification string.
    ambiguous       : True when multiple agents scored within AMBIGUITY_THRESHOLD.
    ambiguous_agents: The tied agents when ambiguous=True.
    """
    selected_agent  : Optional[str]
    candidates      : List[str]
    scores          : Dict[str, float]
    reason          : str
    ambiguous       : bool              = False
    ambiguous_agents: List[str]         = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Router implementation
# ──────────────────────────────────────────────────────────────────────────────

class DynamicRouter(IRouter):
    """
    Concrete IRouter implementation.

    Routing strategy (in priority order)
    -------------------------------------
    1. Delegate to IScheduler.get_ready_agents() to filter dependency-blocked agents.
    2. Exclude agents already completed, failed permanently, or timed-out.
    3. Score each candidate via _score_candidate() (override to customise).
    4. Detect ambiguity: if top-2 scores are within AMBIGUITY_THRESHOLD, log it.
    5. Select highest-scoring candidate; return None if none remain.

    Parameters
    ----------
    scheduler          : IScheduler — dependency-readiness source of truth.
    ambiguity_threshold: float      — max score gap still considered ambiguous.
    """

    AMBIGUITY_THRESHOLD: float = 0.05   # scores within 5 % are "tied"

    def __init__(
        self,
        scheduler: IScheduler,
        ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
    ) -> None:
        self._scheduler           = scheduler
        self._ambiguity_threshold = ambiguity_threshold

    # ── public API (IRouter contract) ─────────────────────────────────────────

    async def select_next_agent(
        self,
        context: SharedContext,
    ) -> Optional[str]:
        """
        Select the next agent to execute.

        Input  : SharedContext
        Output : str (agent name) | None

        Raises
        ------
        ValueError   — no routing resolution possible.
        RuntimeError — context is in an irrecoverable state.
        """
        if context.budget_exhausted:
            logger.warning(
                "Budget exhausted (used=%d / limit=%d). Halting routing.",
                context.tokens_used,
                context.token_budget,
            )
            _emit_event(context, EventType.BUDGET_EXCEEDED, "orchestrator", {
                "tokens_used": context.tokens_used,
                "token_budget": context.token_budget,
            })
            return None

        ready_agents = self._scheduler.get_ready_agents(context)
        if not ready_agents:
            if self._scheduler.is_complete(context):
                logger.info("Scheduler reports all agents complete. Stopping.")
                return None
            logger.warning(
                "No agents are ready but graph is not complete — possible deadlock."
            )
            return None

        decision = self._make_decision(context, ready_agents)
        self._log_decision(context, decision)
        return decision.selected_agent

    # ── internal helpers ──────────────────────────────────────────────────────

    def _make_decision(
        self,
        context: SharedContext,
        ready_agents: List[str],
    ) -> RoutingDecision:
        """Score all ready candidates and build a RoutingDecision."""
        if not ready_agents:
            return RoutingDecision(
                selected_agent=None,
                candidates=[],
                scores={},
                reason="No ready candidates.",
            )

        scores: Dict[str, float] = {
            agent: self._score_candidate(agent, context)
            for agent in ready_agents
        }

        ranked = sorted(scores, key=lambda a: scores[a], reverse=True)
        best   = ranked[0]
        best_score = scores[best]

        # Ambiguity detection
        ambiguous_agents = [
            a for a in ranked[1:]
            if (best_score - scores[a]) <= self._ambiguity_threshold
        ]
        ambiguous = len(ambiguous_agents) > 0

        reason = (
            f"Selected '{best}' (score={best_score:.3f}) from candidates "
            f"{ready_agents}. Scores: {scores}."
        )
        if ambiguous:
            reason += (
                f" AMBIGUOUS: agents {ambiguous_agents} are within "
                f"{self._ambiguity_threshold} of the top score."
            )

        return RoutingDecision(
            selected_agent=best,
            candidates=ready_agents,
            scores=scores,
            reason=reason,
            ambiguous=ambiguous,
            ambiguous_agents=ambiguous_agents,
        )

    def _score_candidate(
        self,
        agent_name: str,
        context: SharedContext,
    ) -> float:
        """
        Heuristic scoring function.  Override in subclasses for custom logic.

        Scoring criteria (additive)
        ---------------------------
        +0.50  — agent has no previous failure record in this run
        +0.30  — agent has zero dependents waiting on it (leaf node = urgent)
        +0.20  — agent name appears as a keyword in the goal string (relevance)
        -0.40  — agent status is FAILED (should not normally appear; safety guard)

        Input  : str, SharedContext
        Output : float in [0.0, 1.0] (approximately)
        """
        score = 0.0

        status = context.agent_statuses.get(agent_name, ExecutionStatus.PENDING)
        if status == ExecutionStatus.FAILED:
            score -= 0.40
        else:
            score += 0.50

        # Agents that other agents are waiting on should run sooner.
        dependents_waiting = sum(
            1
            for deps in context.dependency_graph.values()
            if agent_name in deps
        )
        if dependents_waiting == 0:
            score += 0.30

        # Naive goal-relevance: check if agent name token appears in goal.
        goal_tokens = context.goal.lower().split()
        if any(token in agent_name.lower() for token in goal_tokens):
            score += 0.20

        return max(0.0, score)

    def _log_decision(
        self,
        context: SharedContext,
        decision: RoutingDecision,
    ) -> None:
        """Emit structured log entries and AgentExecutionEvents."""
        event_type = (
            EventType.ROUTING_AMBIGUOUS
            if decision.ambiguous
            else EventType.ROUTING_DECISION
        )
        log_payload = {
            "selected_agent"  : decision.selected_agent,
            "candidates"      : decision.candidates,
            "scores"          : decision.scores,
            "reason"          : decision.reason,
            "ambiguous"       : decision.ambiguous,
            "ambiguous_agents": decision.ambiguous_agents,
        }

        if decision.ambiguous:
            logger.warning("Ambiguous routing detected: %s", log_payload)
        else:
            logger.info("Routing decision: %s", log_payload)

        _emit_event(context, event_type, "orchestrator", log_payload)


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper — emits an AgentExecutionEvent into context metadata sink
# ──────────────────────────────────────────────────────────────────────────────

def _emit_event(
    context   : SharedContext,
    event_type: EventType,
    agent_name: str,
    metadata  : dict,
) -> None:
    """Append an AgentExecutionEvent to context.metadata['_event_sink'] list."""
    sink: list = context.metadata.setdefault("_event_sink", [])
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
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import uuid
    from contracts.models import ExecutionStatus, SharedContext
    from orchestrator.scheduler import DependencyScheduler

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    async def _debug() -> None:
        print("\n" + "=" * 70)
        print("router.py — debug harness")
        print("=" * 70)

        scheduler = DependencyScheduler()
        router    = DynamicRouter(scheduler=scheduler)

        # Scenario: three agents, B depends on A, C is independent
        ctx = SharedContext(
            task_id="debug-router-01",
            goal="fetch analyse report",
            token_budget=50_000,
            available_agents=["fetcher_agent", "analyser_agent", "reporter_agent"],
            dependency_graph={
                "fetcher_agent"  : [],
                "analyser_agent" : ["fetcher_agent"],
                "reporter_agent" : ["analyser_agent"],
            },
            agent_statuses={
                "fetcher_agent"  : ExecutionStatus.PENDING,
                "analyser_agent" : ExecutionStatus.PENDING,
                "reporter_agent" : ExecutionStatus.PENDING,
            },
        )

        # Step 1: nothing completed yet
        agent = await router.select_next_agent(ctx)
        print(f"\n[Step 1] Router selected: {agent}")
        assert agent == "fetcher_agent", f"Expected fetcher_agent, got {agent}"

        # Step 2: mark fetcher done
        ctx.agent_statuses["fetcher_agent"] = ExecutionStatus.COMPLETED
        ctx.completed_agents.append("fetcher_agent")
        agent = await router.select_next_agent(ctx)
        print(f"[Step 2] Router selected: {agent}")
        assert agent == "analyser_agent", f"Expected analyser_agent, got {agent}"

        # Step 3: mark analyser done
        ctx.agent_statuses["analyser_agent"] = ExecutionStatus.COMPLETED
        ctx.completed_agents.append("analyser_agent")
        agent = await router.select_next_agent(ctx)
        print(f"[Step 3] Router selected: {agent}")
        assert agent == "reporter_agent", f"Expected reporter_agent, got {agent}"

        # Step 4: mark all done — expect None
        ctx.agent_statuses["reporter_agent"] = ExecutionStatus.COMPLETED
        ctx.completed_agents.append("reporter_agent")
        agent = await router.select_next_agent(ctx)
        print(f"[Step 4] Router selected (should be None): {agent}")
        assert agent is None

        # Step 5: budget exhausted path
        ctx2 = SharedContext(
            task_id="debug-router-02",
            goal="test",
            token_budget=100,
            tokens_used=100,
            available_agents=["agent_x"],
            dependency_graph={"agent_x": []},
            agent_statuses={"agent_x": ExecutionStatus.PENDING},
        )
        agent = await router.select_next_agent(ctx2)
        print(f"[Step 5] Budget-exhausted result (should be None): {agent}")
        assert agent is None

        print("\n✅  All router assertions passed.")

    asyncio.run(_debug())
