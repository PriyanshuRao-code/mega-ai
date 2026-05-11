"""
orchestrator/scheduler.py
────────────────────────────────────────────────────────────────────────────────
Dependency-graph-aware scheduler for the multi-agent orchestration system.

Responsibility
--------------
Given the dependency_graph stored in SharedContext, determine which agents
are eligible to run right now (all their declared dependencies have
successfully completed) and whether the full graph has been resolved.

Imports
-------
    Standard : collections, logging, typing
    Internal : contracts.models, interfaces.base

Input datatype  : SharedContext
Output datatype : List[str]  (get_ready_agents) | bool  (is_complete)

Possible exceptions
-------------------
    ValueError — raised when a dependency cycle is detected (DFS + coloring).
                 Also raised if an agent in the dependency_graph references
                 an unknown agent name.

Dependencies
------------
    contracts.models.{SharedContext, ExecutionStatus, AgentExecutionEvent, EventType}
    interfaces.base.IScheduler

SOLID notes
-----------
    S — single responsibility: dependency resolution only; no routing, no retry.
    O — _get_topological_order() is a separate protected method; override it
        in subclasses to change scheduling strategy (e.g. priority-weighted).
    L — fulfils IScheduler without extending its contract.
    I — depends only on SharedContext; does not touch BaseAgent internals.
    D — depends on the IScheduler abstraction, not any concrete orchestrator.
"""

from __future__ import annotations

import os as _os, sys as _sys
if __name__ == "__main__":
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import logging
from collections import deque
from typing import Dict, List, Optional, Set

from contracts.models import (
    AgentExecutionEvent,
    EventType,
    ExecutionStatus,
)
from contracts.shared_context import SharedContext
from orchestrator.interfaces import IScheduler

logger = logging.getLogger("orchestrator.scheduler")


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler implementation
# ──────────────────────────────────────────────────────────────────────────────

class DependencyScheduler(IScheduler):
    """
    Topological / Kahn's-algorithm based agent scheduler.

    Algorithm overview
    ------------------
    1. On each call to get_ready_agents():
       a. Validate the dependency graph for cycles (raises ValueError).
       b. Collect agents whose every dependency is in context.completed_agents.
       c. Exclude agents already COMPLETED, FAILED, RUNNING, or TIMEOUT.
       d. Return the filtered eligible set.

    2. is_complete() returns True when every agent in the graph has
       ExecutionStatus.COMPLETED.

    Cycle detection
    ---------------
    Uses iterative DFS with three-colour marking (WHITE/GREY/BLACK).
    Raises ValueError immediately on back-edge discovery.
    """

    # Colour constants for cycle detection
    _WHITE = 0   # not visited
    _GREY  = 1   # in current DFS stack (cycle candidate)
    _BLACK = 2   # fully processed

    # Statuses that mean "this agent is done for this run"
    _TERMINAL_STATUSES: Set[ExecutionStatus] = {
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.TIMEOUT,
        ExecutionStatus.RUNNING,   # already in-flight — don't re-schedule
        ExecutionStatus.SKIPPED,
    }

    # ── IScheduler contract ───────────────────────────────────────────────────

    def get_ready_agents(self, context: SharedContext) -> List[str]:
        """
        Return agents whose dependencies are all satisfied.

        Input  : SharedContext
        Output : List[str] — names of agents ready to execute now.

        Raises
        ------
        ValueError — dependency cycle detected or unknown agent referenced.
        """
        graph = context.dependency_graph
        if not graph:
            logger.debug("Empty dependency graph — nothing to schedule.")
            return []

        self._validate_graph(graph, context.available_agents)
        self._detect_cycles(graph)

        completed: Set[str] = set(context.completed_agents)

        ready: List[str] = []
        for agent, deps in graph.items():
            status = context.agent_statuses.get(agent, ExecutionStatus.PENDING)
            if status in self._TERMINAL_STATUSES:
                continue  # already handled
            if all(dep in completed for dep in deps):
                ready.append(agent)
                logger.debug(
                    "Agent '%s' is READY (deps=%s, completed=%s).",
                    agent, deps, sorted(completed),
                )
            else:
                blocked_by = [d for d in deps if d not in completed]
                logger.debug(
                    "Agent '%s' is BLOCKED by: %s.",
                    agent, blocked_by,
                )
                self._emit_blocked_event(context, agent, blocked_by)

        return ready

    def is_complete(self, context: SharedContext) -> bool:
        """
        True when every agent in the graph has COMPLETED.

        Input  : SharedContext
        Output : bool

        Raises : (none)
        """
        graph = context.dependency_graph
        if not graph:
            return True

        for agent in graph:
            status = context.agent_statuses.get(agent, ExecutionStatus.PENDING)
            # We want to know if every agent is finished (COMPLETED, FAILED, TIMEOUT, or SKIPPED)
            # but NOT RUNNING.
            is_terminal = (
                status in self._TERMINAL_STATUSES 
                and status != ExecutionStatus.RUNNING
            )
            if not is_terminal:
                logger.debug(
                    "Graph not complete — '%s' is %s.", agent, status.value
                )
                return False

        return True

        logger.info("Dependency graph fully resolved — all agents COMPLETED.")
        return True

    # ── internal helpers ──────────────────────────────────────────────────────

    def _validate_graph(
        self,
        graph: Dict[str, List[str]],
        available_agents: List[str],
    ) -> None:
        """
        Ensure every dependency reference names a known agent.

        Raises
        ------
        ValueError — if a dependency name is not in available_agents.
        """
        known: Set[str] = set(available_agents) | set(graph.keys())
        for agent, deps in graph.items():
            for dep in deps:
                if dep not in known:
                    raise ValueError(
                        f"Dependency '{dep}' declared by '{agent}' "
                        f"is not a known agent. Known: {sorted(known)}"
                    )

    def _detect_cycles(self, graph: Dict[str, List[str]]) -> None:
        """
        Iterative three-colour DFS cycle detection.

        Raises
        ------
        ValueError — if a back-edge (cycle) is found.
        """
        colour: Dict[str, int] = {node: self._WHITE for node in graph}

        for start in graph:
            if colour[start] != self._WHITE:
                continue

            # Iterative DFS: stack holds (node, iterator_over_neighbours)
            stack = [(start, iter(graph.get(start, [])))]
            colour[start] = self._GREY

            while stack:
                node, neighbours = stack[-1]
                try:
                    neighbour = next(neighbours)
                    if neighbour not in colour:
                        # Neighbour only appears as a dep, not a graph key
                        continue
                    if colour[neighbour] == self._GREY:
                        cycle_path = [n for n, _ in stack] + [neighbour]
                        raise ValueError(
                            f"Dependency cycle detected: {' → '.join(cycle_path)}"
                        )
                    if colour[neighbour] == self._WHITE:
                        colour[neighbour] = self._GREY
                        stack.append(
                            (neighbour, iter(graph.get(neighbour, [])))
                        )
                except StopIteration:
                    colour[node] = self._BLACK
                    stack.pop()

    def _get_topological_order(
        self,
        graph: Dict[str, List[str]],
    ) -> List[str]:
        """
        Return a valid topological ordering of all agents (Kahn's algorithm).
        Used for logging and introspection; not required for get_ready_agents().

        Output : List[str] — topologically sorted agent names.
        Raises : ValueError — if a cycle exists (delegates to _detect_cycles).
        """
        self._detect_cycles(graph)
        in_degree: Dict[str, int] = {node: 0 for node in graph}
        for deps in graph.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] = in_degree.get(dep, 0)  # already 0

        # Rebuild in-degree from forward edges
        reverse_in: Dict[str, int] = {node: 0 for node in graph}
        for node, deps in graph.items():
            for dep in deps:
                reverse_in[node] = reverse_in.get(node, 0)
                # node depends on dep → dep must come before node
                # so dep has lower "in-degree" in Kahn's sense here
        # Kahn's uses adjacency as "node → successors"; our graph is inverted
        # (node → prerequisites). Flip it.
        successors: Dict[str, List[str]] = {n: [] for n in graph}
        in_deg: Dict[str, int]           = {n: 0  for n in graph}
        for node, prereqs in graph.items():
            for pre in prereqs:
                if pre in successors:
                    successors[pre].append(node)
                    in_deg[node] += 1

        queue  = deque(n for n in graph if in_deg[n] == 0)
        result: List[str] = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for successor in successors.get(node, []):
                in_deg[successor] -= 1
                if in_deg[successor] == 0:
                    queue.append(successor)

        if len(result) != len(graph):
            raise ValueError("Cycle detected during topological sort.")
        return result

    @staticmethod
    def _emit_blocked_event(
        context    : SharedContext,
        agent_name : str,
        blocked_by : List[str],
    ) -> None:
        """Append a DEPENDENCY_BLOCKED event to the context's event sink."""
        sink: list = context.metadata.setdefault("_event_sink", [])
        sink.append(
            AgentExecutionEvent(
                event_type=EventType.DEPENDENCY_BLOCKED,
                agent_name=agent_name,
                task_id=context.task_id,
                metadata={"blocked_by": blocked_by},
            )
        )


# ──────────────────────────────────────────────────────────────────────────────
# Self-contained debug harness
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from contracts.models import ExecutionStatus, SharedContext

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    def _debug() -> None:
        print("\n" + "=" * 70)
        print("scheduler.py — debug harness")
        print("=" * 70)

        scheduler = DependencyScheduler()

        # Graph: A → B → D, A → C → D  (diamond)
        graph = {
            "agent_a": [],
            "agent_b": ["agent_a"],
            "agent_c": ["agent_a"],
            "agent_d": ["agent_b", "agent_c"],
        }
        all_agents = list(graph.keys())

        ctx = SharedContext(
            task_id="debug-sched-01",
            goal="test scheduling",
            token_budget=50_000,
            available_agents=all_agents,
            dependency_graph=graph,
            agent_statuses={a: ExecutionStatus.PENDING for a in all_agents},
        )

        # Step 1: nothing done → only A ready
        ready = scheduler.get_ready_agents(ctx)
        print(f"\n[Step 1] Ready (expect [agent_a]): {ready}")
        assert ready == ["agent_a"], ready

        # Step 2: A done → B and C ready
        ctx.completed_agents.append("agent_a")
        ctx.agent_statuses["agent_a"] = ExecutionStatus.COMPLETED
        ready = scheduler.get_ready_agents(ctx)
        print(f"[Step 2] Ready (expect [agent_b, agent_c]): {sorted(ready)}")
        assert sorted(ready) == ["agent_b", "agent_c"], ready

        # Step 3: B done, C still pending → D still blocked
        ctx.completed_agents.append("agent_b")
        ctx.agent_statuses["agent_b"] = ExecutionStatus.COMPLETED
        ready = scheduler.get_ready_agents(ctx)
        print(f"[Step 3] Ready (expect [agent_c]): {ready}")
        assert ready == ["agent_c"], ready

        # Step 4: C done → D ready
        ctx.completed_agents.append("agent_c")
        ctx.agent_statuses["agent_c"] = ExecutionStatus.COMPLETED
        ready = scheduler.get_ready_agents(ctx)
        print(f"[Step 4] Ready (expect [agent_d]): {ready}")
        assert ready == ["agent_d"], ready

        # Step 5: D done → graph complete
        ctx.completed_agents.append("agent_d")
        ctx.agent_statuses["agent_d"] = ExecutionStatus.COMPLETED
        print(f"[Step 5] Is complete (expect True): {scheduler.is_complete(ctx)}")
        assert scheduler.is_complete(ctx)

        # Step 6: cycle detection
        cyclic_ctx = SharedContext(
            task_id="debug-sched-02",
            goal="cycle test",
            token_budget=50_000,
            available_agents=["x", "y", "z"],
            dependency_graph={"x": ["z"], "y": ["x"], "z": ["y"]},
            agent_statuses={"x": ExecutionStatus.PENDING,
                            "y": ExecutionStatus.PENDING,
                            "z": ExecutionStatus.PENDING},
        )
        try:
            scheduler.get_ready_agents(cyclic_ctx)
            print("❌ No exception raised — test FAILED.")
        except ValueError as exc:
            print(f"[Step 6] Cycle correctly detected: {exc}")

        # Step 7: topological order
        topo = scheduler._get_topological_order(graph)
        print(f"[Step 7] Topological order: {topo}")

        print("\n✅  All scheduler assertions passed.")

    _debug()
