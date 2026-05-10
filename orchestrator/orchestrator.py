"""
orchestrator/orchestrator.py
────────────────────────────────────────────────────────────────────────────────
Top-level orchestration controller for the multi-agent system.

Responsibility
--------------
Coordinate the full lifecycle of a multi-agent run:
  1. Register agents and build the dependency graph on SharedContext.
  2. Enter a scheduling loop: ask IRouter for the next agent.
  3. Gate execution through IScheduler (dependency readiness).
  4. Execute via IRetryManager (with back-off and timeout).
  5. Accumulate ToolResponse results into SharedContext.
  6. Detect and log policy violations.
  7. Enforce the context token budget.
  8. Produce a complete ExecutionEvent log as output.

Imports
-------
    Standard : asyncio, datetime, logging, typing, uuid
    Internal : contracts.models, interfaces.base,
               orchestrator.router, orchestrator.scheduler,
               orchestrator.retry_manager

Input datatype  : SharedContext
Output datatype : tuple[SharedContext, ExecutionEvent]

Possible exceptions
-------------------
    RuntimeError — unrecoverable orchestration failure (e.g. infinite loop guard).
    ValueError   — dependency cycle / bad graph detected by scheduler.

Dependencies
------------
    IRouter, IScheduler, IRetryManager — all injected (constructor DI).
    SharedContext, ExecutionEvent, AgentExecutionEvent — contracts layer.

SOLID notes
-----------
    S — orchestration loop only; no routing logic, no back-off math.
    O — extend by subclassing and overriding _post_agent_hook().
    L — fulfils IOrchestrator without changing the contract.
    I — depends only on the three narrow strategy interfaces.
    D — every dependency injected; no internal `new` / instantiation.
"""

from __future__ import annotations

import os as _os, sys as _sys
if __name__ == "__main__":
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from contracts.models import (
    AgentExecutionEvent,
    EventType,
    ExecutionEvent,
    ExecutionStatus,
    PolicyViolation,
    SharedContext,
    ToolResponse,
)
from interfaces.base import BaseAgent, IOrchestrator, IRetryManager, IRouter, IScheduler
from orchestrator.retry_manager import ExponentialRetryManager, RetryPolicy
from orchestrator.router import DynamicRouter
from orchestrator.scheduler import DependencyScheduler

logger = logging.getLogger("orchestrator.orchestrator")


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class MultiAgentOrchestrator(IOrchestrator):
    """
    Production-grade orchestration controller.

    Constructor parameters
    ----------------------
    agents          : List of BaseAgent instances to register.
    router          : IRouter — selects the next agent to run.
    scheduler       : IScheduler — dependency-graph readiness checks.
    retry_manager   : IRetryManager — wraps execution with retry/timeout.
    max_iterations  : Hard loop-guard to prevent infinite cycles (default 1000).
    policy_rules    : Callable list for custom policy checks per ToolResponse.
    """

    _DEFAULT_MAX_ITERATIONS = 1_000

    def __init__(
        self,
        agents         : List[BaseAgent],
        router         : IRouter,
        scheduler      : IScheduler,
        retry_manager  : IRetryManager,
        max_iterations : int = _DEFAULT_MAX_ITERATIONS,
        policy_rules   : Optional[List] = None,
    ) -> None:
        self._agents         : Dict[str, BaseAgent] = {a.name: a for a in agents}
        self._router         = router
        self._scheduler      = scheduler
        self._retry_manager  = retry_manager
        self._max_iterations = max_iterations
        self._policy_rules   = policy_rules or []

    # ── IOrchestrator contract ────────────────────────────────────────────────

    async def run(
        self, context: SharedContext
    ) -> tuple[SharedContext, ExecutionEvent]:
        """
        Execute the full agent graph and return updated state + logs.

        Input  : SharedContext
        Output : tuple[SharedContext, ExecutionEvent]

        Raises
        ------
        RuntimeError — if the loop guard (max_iterations) is exceeded.
        ValueError   — if the dependency graph contains a cycle.
        """
        run_id     = context.task_id
        started_at = datetime.utcnow()
        event_sink : List[AgentExecutionEvent] = []

        logger.info("Orchestration START — task_id=%s goal='%s'", run_id, context.goal)
        self._bootstrap_context(context)

        final_status   = ExecutionStatus.COMPLETED
        terminal_error : Optional[str] = None
        iterations     = 0

        try:
            while iterations < self._max_iterations:
                iterations += 1

                # ── budget guard ─────────────────────────────────────────────
                if context.budget_exhausted:
                    logger.warning(
                        "[%s] Token budget exhausted (%d/%d). Stopping.",
                        run_id, context.tokens_used, context.token_budget,
                    )
                    self._append_event(
                        event_sink, context, EventType.BUDGET_EXCEEDED, "orchestrator",
                        {"tokens_used": context.tokens_used,
                         "token_budget": context.token_budget},
                    )
                    final_status = ExecutionStatus.FAILED
                    terminal_error = "Token budget exhausted."
                    break

                # ── scheduler completion check ────────────────────────────────
                if self._scheduler.is_complete(context):
                    logger.info("[%s] All agents complete. Stopping.", run_id)
                    break

                # ── routing ───────────────────────────────────────────────────
                next_agent_name = await self._router.select_next_agent(context)
                if next_agent_name is None:
                    logger.info(
                        "[%s] Router returned None — stopping orchestration.", run_id
                    )
                    break

                agent = self._agents.get(next_agent_name)
                if agent is None:
                    logger.error(
                        "[%s] Router selected unknown agent '%s'. Skipping.",
                        run_id, next_agent_name,
                    )
                    context.agent_statuses[next_agent_name] = ExecutionStatus.SKIPPED
                    continue

                # ── pre-flight validation ─────────────────────────────────────
                if not await agent.validate(context):
                    logger.warning(
                        "[%s] Agent '%s' failed pre-flight validation. Skipping.",
                        run_id, next_agent_name,
                    )
                    context.agent_statuses[next_agent_name] = ExecutionStatus.SKIPPED
                    self._append_event(
                        event_sink, context, EventType.AGENT_FAILED,
                        next_agent_name, {"reason": "pre-flight validation failed"},
                    )
                    continue

                # ── execution ─────────────────────────────────────────────────
                logger.info(
                    "[%s] Executing agent '%s' (iteration %d).",
                    run_id, next_agent_name, iterations,
                )
                response: ToolResponse = await self._retry_manager.execute_with_retry(
                    agent, context, event_sink
                )

                # ── record result ─────────────────────────────────────────────
                context.record_agent_output(response)

                # ── policy checks ─────────────────────────────────────────────
                violations = self._run_policy_checks(response, context)
                if violations:
                    for v in violations:
                        context.policy_flags.append(v)
                        self._append_event(
                            event_sink, context, EventType.POLICY_VIOLATION,
                            response.agent_name,
                            {"rule": v.rule, "severity": v.severity,
                             "description": v.description},
                        )
                    logger.warning(
                        "[%s] %d policy violation(s) from agent '%s'.",
                        run_id, len(violations), next_agent_name,
                    )

                # ── post-agent hook (override in subclass) ────────────────────
                await self._post_agent_hook(response, context, event_sink)

                # ── scheduler cycle event ─────────────────────────────────────
                self._append_event(
                    event_sink, context, EventType.SCHEDULER_CYCLE, "orchestrator",
                    {"iteration": iterations,
                     "completed": list(context.completed_agents),
                     "tokens_used": context.tokens_used},
                )

            else:
                # Loop guard triggered
                msg = (
                    f"Orchestration loop exceeded max_iterations={self._max_iterations}."
                )
                logger.error("[%s] %s", run_id, msg)
                final_status  = ExecutionStatus.FAILED
                terminal_error = msg
                raise RuntimeError(msg)

        except (ValueError, RuntimeError) as exc:
            final_status   = ExecutionStatus.FAILED
            terminal_error = str(exc)
            logger.exception("[%s] Orchestration terminated with error.", run_id)
        except Exception as exc:  # noqa: BLE001
            final_status   = ExecutionStatus.FAILED
            terminal_error = f"Unexpected: {exc}"
            logger.exception("[%s] Unhandled exception in orchestration.", run_id)

        finished_at = datetime.utcnow()

        # Merge any events emitted into context.metadata['_event_sink'] by router/scheduler
        extra_events: List[AgentExecutionEvent] = context.metadata.pop("_event_sink", [])
        all_events = extra_events + event_sink  # router/scheduler events come first

        self._append_event(
            all_events, context, EventType.ORCHESTRATION_DONE, "orchestrator",
            {
                "status"          : final_status.value,
                "iterations"      : iterations,
                "total_tokens"    : context.tokens_used,
                "completed_agents": list(context.completed_agents),
                "error"           : terminal_error,
            },
        )

        execution_event = ExecutionEvent(
            task_id           = run_id,
            status            = final_status,
            agent_events      = all_events,
            policy_violations = list(context.policy_flags),
            total_tokens_used = context.tokens_used,
            started_at        = started_at,
            finished_at       = finished_at,
            error             = terminal_error,
        )

        logger.info(
            "Orchestration END — task_id=%s status=%s tokens=%d iterations=%d",
            run_id, final_status.value, context.tokens_used, iterations,
        )
        return context, execution_event

    # ── override points ───────────────────────────────────────────────────────

    async def _post_agent_hook(
        self,
        response   : ToolResponse,
        context    : SharedContext,
        event_sink : List[AgentExecutionEvent],
    ) -> None:
        """
        Called after every agent execution.  No-op by default.
        Override in subclasses for cross-cutting concerns
        (e.g. checkpoint saves, telemetry pushes, context summarisation).
        """

    # ── private helpers ───────────────────────────────────────────────────────

    def _bootstrap_context(self, context: SharedContext) -> None:
        """
        Ensure context.available_agents and agent_statuses are initialised
        from the registered agent registry if they are empty.
        """
        if not context.available_agents:
            context.available_agents = list(self._agents.keys())
            logger.debug(
                "Bootstrapped available_agents from registry: %s",
                context.available_agents,
            )

        for agent_name in context.available_agents:
            context.agent_statuses.setdefault(agent_name, ExecutionStatus.PENDING)

        if not context.dependency_graph:
            # Flat graph: no inter-dependencies
            context.dependency_graph = {
                name: [] for name in context.available_agents
            }
            logger.debug(
                "Bootstrapped flat dependency_graph: %s", context.dependency_graph
            )

    def _run_policy_checks(
        self,
        response: ToolResponse,
        context : SharedContext,
    ) -> List[PolicyViolation]:
        """
        Apply every registered policy rule to the response.

        Input  : ToolResponse, SharedContext
        Output : List[PolicyViolation]
        """
        violations: List[PolicyViolation] = []
        for rule_fn in self._policy_rules:
            try:
                result = rule_fn(response, context)
                if result is not None:
                    violations.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.error("Policy rule %s raised %s: %s", rule_fn, type(exc).__name__, exc)
        return violations

    @staticmethod
    def _append_event(
        sink      : List[AgentExecutionEvent],
        context   : SharedContext,
        event_type: EventType,
        agent_name: str,
        metadata  : dict,
    ) -> None:
        sink.append(
            AgentExecutionEvent(
                event_type=event_type,
                agent_name=agent_name,
                task_id=context.task_id,
                metadata=metadata,
            )
        )


# ──────────────────────────────────────────────────────────────────────────────
# Factory helper
# ──────────────────────────────────────────────────────────────────────────────

def build_orchestrator(
    agents       : List[BaseAgent],
    retry_policy : RetryPolicy = RetryPolicy(),
    policy_rules : Optional[List] = None,
    max_iterations: int = MultiAgentOrchestrator._DEFAULT_MAX_ITERATIONS,
) -> MultiAgentOrchestrator:
    """
    Convenience factory — assembles the default orchestrator stack.

    Input
    -----
    agents        : List[BaseAgent]  — agents to register.
    retry_policy  : RetryPolicy      — retry/timeout configuration.
    policy_rules  : list of callables(ToolResponse, SharedContext) → Optional[PolicyViolation]
    max_iterations: int              — loop guard.

    Output
    ------
    MultiAgentOrchestrator — fully wired, ready to call .run(context).
    """
    scheduler     = DependencyScheduler()
    router        = DynamicRouter(scheduler=scheduler)
    retry_manager = ExponentialRetryManager(policy=retry_policy)
    return MultiAgentOrchestrator(
        agents=agents,
        router=router,
        scheduler=scheduler,
        retry_manager=retry_manager,
        max_iterations=max_iterations,
        policy_rules=policy_rules or [],
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
    from interfaces.base import BaseAgent

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # ── Stub agents ───────────────────────────────────────────────────────────

    class FetcherAgent(BaseAgent):
        @property
        def name(self)       -> str: return "fetcher"
        @property
        def max_tokens(self) -> int: return 1_000
        async def execute(self, context: SharedContext) -> ToolResponse:
            await asyncio.sleep(0.01)
            return ToolResponse(agent_name=self.name, output={"data": [1, 2, 3]},
                                tokens_used=150, success=True)

    class AnalyserAgent(BaseAgent):
        @property
        def name(self)       -> str: return "analyser"
        @property
        def max_tokens(self) -> int: return 2_000
        async def execute(self, context: SharedContext) -> ToolResponse:
            fetcher_out = context.agent_outputs.get("fetcher")
            data = fetcher_out.output["data"] if fetcher_out else []
            await asyncio.sleep(0.01)
            return ToolResponse(agent_name=self.name,
                                output={"sum": sum(data), "count": len(data)},
                                tokens_used=300, success=True)

    class ReporterAgent(BaseAgent):
        @property
        def name(self)       -> str: return "reporter"
        @property
        def max_tokens(self) -> int: return 500
        async def execute(self, context: SharedContext) -> ToolResponse:
            analysis = context.agent_outputs.get("analyser")
            summary  = analysis.output if analysis else {}
            await asyncio.sleep(0.01)
            return ToolResponse(agent_name=self.name,
                                output=f"Report: {summary}",
                                tokens_used=80, success=True)

    async def _debug() -> None:
        print("\n" + "=" * 70)
        print("orchestrator.py — debug harness")
        print("=" * 70)

        agents = [FetcherAgent(), AnalyserAgent(), ReporterAgent()]
        orch   = build_orchestrator(
            agents=agents,
            retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0.01,
                                     timeout_s=5.0, jitter=False),
        )

        ctx = SharedContext(
            task_id="debug-orch-01",
            goal="fetch analyse report",
            token_budget=10_000,
            available_agents=["fetcher", "analyser", "reporter"],
            dependency_graph={
                "fetcher"  : [],
                "analyser" : ["fetcher"],
                "reporter" : ["analyser"],
            },
        )

        final_ctx, exec_event = await orch.run(ctx)

        print(f"\nStatus        : {exec_event.status}")
        print(f"Total tokens  : {exec_event.total_tokens_used}")
        print(f"Completed     : {final_ctx.completed_agents}")
        print(f"Events logged : {len(exec_event.agent_events)}")
        print(f"Violations    : {len(exec_event.policy_violations)}")
        print("\nAgent outputs:")
        for name, resp in final_ctx.agent_outputs.items():
            print(f"  {name}: {resp.output}")
        print("\nEvent log (type | agent):")
        for ev in exec_event.agent_events:
            print(f"  {ev.event_type.value:<28} | {ev.agent_name}")

        assert exec_event.status == ExecutionStatus.COMPLETED
        assert "reporter" in final_ctx.completed_agents
        print("\n✅  All orchestrator assertions passed.")

    asyncio.run(_debug())
