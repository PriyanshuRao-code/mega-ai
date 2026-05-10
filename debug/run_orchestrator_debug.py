"""
debug/run_orchestrator_debug.py
────────────────────────────────────────────────────────────────────────────────
End-to-end debug runner for the full multi-agent orchestration stack.

Runs FIVE distinct scenarios and prints a rich structured report for each:

  Scenario 1 — Happy path: linear chain A → B → C, all succeed.
  Scenario 2 — Diamond dependency: A → {B, C} → D, parallel readiness.
  Scenario 3 — Flaky agent: B fails twice then recovers (retry validation).
  Scenario 4 — Budget enforcement: budget too small to complete the graph.
  Scenario 5 — Policy violation: reporter agent triggers a custom rule.

Imports
-------
    Standard : asyncio, logging, sys, textwrap, datetime
    Internal : contracts.models, interfaces.base, orchestrator.*

Input  : (none — run directly: `python debug/run_orchestrator_debug.py`)
Output : console log + PASS/FAIL summary

Possible exceptions
-------------------
    AssertionError — raised on scenario failure (test guard).
    SystemExit     — exits with code 1 if any scenario fails.

Dependencies
------------
    All four orchestrator modules + contracts + interfaces.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import textwrap
import uuid
from datetime import datetime
from typing import List, Optional

# ── path setup (run from repo root or debug/) ─────────────────────────────────
sys.path.insert(0, __file__.rsplit("/debug", 1)[0])   # repo root

from contracts.models import (
    ExecutionStatus,
    PolicyViolation,
    SharedContext,
    ToolResponse,
)
from interfaces.base import BaseAgent
from orchestrator.orchestrator import build_orchestrator
from orchestrator.retry_manager import RetryPolicy

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,   # quiet during debug — flip to INFO/DEBUG if needed
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)

# ──────────────────────────────────────────────────────────────────────────────
# Reusable stub agent base
# ──────────────────────────────────────────────────────────────────────────────

class SimpleAgent(BaseAgent):
    """
    Configurable stub agent for debug scenarios.

    Parameters
    ----------
    agent_name     : str — unique identifier.
    tokens_cost    : int — tokens reported consumed per call.
    fail_times     : int — how many times to return success=False before succeeding.
    sleep_s        : float — simulated latency per call.
    output_payload : any — value placed in ToolResponse.output on success.
    """

    def __init__(
        self,
        agent_name     : str,
        tokens_cost    : int   = 100,
        fail_times     : int   = 0,
        sleep_s        : float = 0.005,
        output_payload : object = None,
    ) -> None:
        self._name     = agent_name
        self._cost     = tokens_cost
        self._fails    = fail_times
        self._calls    = 0
        self._sleep    = sleep_s
        self._payload  = output_payload or f"{agent_name}::done"

    @property
    def name(self) -> str:        return self._name
    @property
    def max_tokens(self) -> int:  return 2_000

    async def execute(self, context: SharedContext) -> ToolResponse:
        await asyncio.sleep(self._sleep)
        self._calls += 1
        if self._calls <= self._fails:
            return ToolResponse(
                agent_name=self.name,
                output=None,
                tokens_used=10,
                success=False,
                error=f"Simulated failure #{self._calls}",
            )
        return ToolResponse(
            agent_name=self.name,
            output=self._payload,
            tokens_used=self._cost,
            success=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helper — print scenario header / footer
# ──────────────────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    bar = "─" * 68
    print(f"\n┌{bar}┐")
    print(f"│  {title:<66}│")
    print(f"└{bar}┘")


def _report(ctx: SharedContext, evt) -> None:
    """Print a compact structured report for one scenario."""
    print(f"  Status        : {evt.status.value}")
    print(f"  Tokens used   : {evt.total_tokens_used:,} / {ctx.token_budget:,}")
    print(f"  Completed     : {ctx.completed_agents}")
    print(f"  Events        : {len(evt.agent_events)}")
    print(f"  Violations    : {len(evt.policy_violations)}")
    if evt.error:
        print(f"  Error         : {evt.error}")
    print(f"  Duration      : {(evt.finished_at - evt.started_at).total_seconds():.3f}s")
    if ctx.agent_outputs:
        print("  Outputs:")
        for name, resp in ctx.agent_outputs.items():
            truncated = str(resp.output)[:60]
            print(f"    {name:<18}: {truncated}")


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Happy path linear chain
# ──────────────────────────────────────────────────────────────────────────────

async def scenario_happy_path() -> bool:
    _header("Scenario 1 — Happy path: linear chain  A → B → C")

    agents = [
        SimpleAgent("agent_a", tokens_cost=200),
        SimpleAgent("agent_b", tokens_cost=300),
        SimpleAgent("agent_c", tokens_cost=150),
    ]
    orch = build_orchestrator(
        agents=agents,
        retry_policy=RetryPolicy(max_attempts=1, timeout_s=5.0, jitter=False),
    )
    ctx = SharedContext(
        task_id="sc1-happy",
        goal="agent_a agent_b agent_c",
        token_budget=10_000,
        available_agents=["agent_a", "agent_b", "agent_c"],
        dependency_graph={
            "agent_a": [],
            "agent_b": ["agent_a"],
            "agent_c": ["agent_b"],
        },
    )

    final_ctx, evt = await orch.run(ctx)
    _report(final_ctx, evt)

    passed = (
        evt.status == ExecutionStatus.COMPLETED
        and set(final_ctx.completed_agents) == {"agent_a", "agent_b", "agent_c"}
    )
    print(f"  → {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 2 — Diamond dependency graph
# ──────────────────────────────────────────────────────────────────────────────

async def scenario_diamond() -> bool:
    _header("Scenario 2 — Diamond graph:  A → {B, C} → D")

    agents = [
        SimpleAgent("a", tokens_cost=100),
        SimpleAgent("b", tokens_cost=200),
        SimpleAgent("c", tokens_cost=200),
        SimpleAgent("d", tokens_cost=100),
    ]
    orch = build_orchestrator(
        agents=agents,
        retry_policy=RetryPolicy(max_attempts=1, timeout_s=5.0, jitter=False),
    )
    ctx = SharedContext(
        task_id="sc2-diamond",
        goal="a b c d",
        token_budget=10_000,
        available_agents=["a", "b", "c", "d"],
        dependency_graph={
            "a": [],
            "b": ["a"],
            "c": ["a"],
            "d": ["b", "c"],
        },
    )

    final_ctx, evt = await orch.run(ctx)
    _report(final_ctx, evt)

    passed = (
        evt.status == ExecutionStatus.COMPLETED
        and set(final_ctx.completed_agents) == {"a", "b", "c", "d"}
    )
    print(f"  → {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 3 — Flaky agent: fails twice then succeeds
# ──────────────────────────────────────────────────────────────────────────────

async def scenario_retry() -> bool:
    _header("Scenario 3 — Flaky agent:  fetcher fails ×2, then recovers")

    agents = [
        SimpleAgent("fetcher", tokens_cost=100, fail_times=2),
        SimpleAgent("processor", tokens_cost=200),
    ]
    orch = build_orchestrator(
        agents=agents,
        retry_policy=RetryPolicy(
            max_attempts=4,
            base_delay_s=0.01,
            max_delay_s=0.05,
            timeout_s=5.0,
            jitter=False,
        ),
    )
    ctx = SharedContext(
        task_id="sc3-retry",
        goal="fetcher processor",
        token_budget=10_000,
        available_agents=["fetcher", "processor"],
        dependency_graph={"fetcher": [], "processor": ["fetcher"]},
    )

    final_ctx, evt = await orch.run(ctx)
    _report(final_ctx, evt)

    # Count AGENT_RETRYING events for the fetcher
    retry_events = [
        e for e in evt.agent_events
        if e.event_type.value == "AGENT_RETRYING" and e.agent_name == "fetcher"
    ]
    print(f"  Retry events for fetcher: {len(retry_events)}")

    passed = (
        evt.status == ExecutionStatus.COMPLETED
        and set(final_ctx.completed_agents) == {"fetcher", "processor"}
        and len(retry_events) == 2
    )
    print(f"  → {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 4 — Budget enforcement
# ──────────────────────────────────────────────────────────────────────────────

async def scenario_budget() -> bool:
    _header("Scenario 4 — Budget enforcement: limit set below total cost")

    agents = [
        SimpleAgent("step1", tokens_cost=400),
        SimpleAgent("step2", tokens_cost=400),
        SimpleAgent("step3", tokens_cost=400),
    ]
    orch = build_orchestrator(
        agents=agents,
        retry_policy=RetryPolicy(max_attempts=1, timeout_s=5.0, jitter=False),
    )
    ctx = SharedContext(
        task_id="sc4-budget",
        goal="step1 step2 step3",
        token_budget=600,   # only enough for ~1.5 agents
        available_agents=["step1", "step2", "step3"],
        dependency_graph={
            "step1": [],
            "step2": ["step1"],
            "step3": ["step2"],
        },
    )

    final_ctx, evt = await orch.run(ctx)
    _report(final_ctx, evt)

    budget_event = any(
        e.event_type.value == "BUDGET_EXCEEDED" for e in evt.agent_events
    )
    passed = (
        evt.status == ExecutionStatus.FAILED
        and budget_event
        and final_ctx.tokens_used >= 400   # at least one agent ran
    )
    print(f"  Budget event fired: {budget_event}")
    print(f"  → {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 5 — Custom policy rule fires on reporter output
# ──────────────────────────────────────────────────────────────────────────────

async def scenario_policy_violation() -> bool:
    _header("Scenario 5 — Policy violation: reporter emits forbidden keyword")

    class ReporterAgent(BaseAgent):
        @property
        def name(self) -> str: return "reporter"
        @property
        def max_tokens(self) -> int: return 500
        async def execute(self, context: SharedContext) -> ToolResponse:
            return ToolResponse(
                agent_name=self.name,
                output="RESTRICTED: classified content detected",
                tokens_used=50,
                success=True,
            )

    def forbidden_keyword_rule(
        response: ToolResponse, context: SharedContext
    ) -> Optional[PolicyViolation]:
        if isinstance(response.output, str) and "RESTRICTED" in response.output:
            return PolicyViolation(
                rule="forbidden_keyword",
                severity="high",
                agent_name=response.agent_name,
                description="Output contains forbidden keyword 'RESTRICTED'.",
            )
        return None

    agents = [SimpleAgent("collector", tokens_cost=80), ReporterAgent()]
    orch   = build_orchestrator(
        agents=agents,
        retry_policy=RetryPolicy(max_attempts=1, timeout_s=5.0, jitter=False),
        policy_rules=[forbidden_keyword_rule],
    )
    ctx = SharedContext(
        task_id="sc5-policy",
        goal="collector reporter",
        token_budget=10_000,
        available_agents=["collector", "reporter"],
        dependency_graph={"collector": [], "reporter": ["collector"]},
    )

    final_ctx, evt = await orch.run(ctx)
    _report(final_ctx, evt)

    violation_events = [
        e for e in evt.agent_events if e.event_type.value == "POLICY_VIOLATION"
    ]
    passed = (
        len(evt.policy_violations) == 1
        and evt.policy_violations[0].rule == "forbidden_keyword"
        and len(violation_events) == 1
    )
    print(f"  Violations captured: {[v.rule for v in evt.policy_violations]}")
    print(f"  → {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    banner = textwrap.dedent("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║         Multi-Agent Orchestrator — End-to-End Debug Runner          ║
    ║                                                                      ║
    ║  Covers: routing · scheduling · retry · budget · policy violations  ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    print(banner)

    results: dict[str, bool] = {}

    for scenario_fn in [
        scenario_happy_path,
        scenario_diamond,
        scenario_retry,
        scenario_budget,
        scenario_policy_violation,
    ]:
        try:
            results[scenario_fn.__name__] = await scenario_fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ EXCEPTION: {exc}")
            results[scenario_fn.__name__] = False

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  SUMMARY")
    print("═" * 70)
    all_passed = True
    for name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {name}")
        if not passed:
            all_passed = False

    print("═" * 70)
    if all_passed:
        print("  🎉  ALL SCENARIOS PASSED")
    else:
        print("  💥  ONE OR MORE SCENARIOS FAILED")
    print()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
