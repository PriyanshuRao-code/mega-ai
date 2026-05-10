"""
agents/decomposition_agent.py
==============================
DecompositionAgent — breaks a top-level query into a dependency-ordered
graph of typed subtasks.

SOLID Alignment:
  - (S) Responsible only for decomposition logic
  - (O) Extend classification strategies without modifying this class
  - (L) Safe to use wherever BaseAgent[SharedContext, DecompositionResult] is expected
  - (D) Depends on interfaces/contracts, not concrete agents

Imports (external):
  stdlib  : logging, re, uuid, collections
  local   : interfaces.base_agent, contracts.shared_context,
            contracts.agent_contracts

Input:
  SharedContext
    .query               — top-level user query (required)
    .conversation_history — optional prior turns used for context
    .metadata             — optional hints: 'max_subtasks', 'categories'

Output:
  DecompositionResult
    .subtasks        — list of SubTask (each with category + dependencies)
    .dependency_edges — directed edges forming the DAG
    .execution_order  — topologically sorted task_id list
    .metadata         — diagnostics

Exceptions:
  AgentValidationError  : query missing or context malformed
  AgentExecutionError   : decomposition or topo-sort fails
  ContractValidationError: output invariant violation (no subtasks)

Dependencies:
  interfaces.base_agent.BaseAgent
  contracts.shared_context.SharedContext
  contracts.agent_contracts.DecompositionResult, SubTask, DependencyEdge, TaskStatus
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

from interfaces.base_agent import BaseAgent, AgentExecutionError, AgentValidationError
from contracts.shared_context import SharedContext
from contracts.agent_contracts import (
    ContractValidationError,
    DecompositionResult,
    DependencyEdge,
    SubTask,
    TaskStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category classifier helpers
# ---------------------------------------------------------------------------

# Maps regex patterns → category label (first match wins, order matters)
_CATEGORY_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(find|search|look up|retrieve|fetch|get)\b", re.I), "retrieval"),
    (re.compile(r"\b(summarize|compress|shorten|condense)\b",    re.I), "compression"),
    (re.compile(r"\b(compare|contrast|evaluate|assess|critique)\b", re.I), "critique"),
    (re.compile(r"\b(combine|merge|synthesize|integrate|unify)\b", re.I), "synthesis"),
    (re.compile(r"\b(reason|infer|deduce|calculate|compute)\b",  re.I), "reasoning"),
]

_DEFAULT_CATEGORY = "general"


def _classify(description: str) -> str:
    for pattern, category in _CATEGORY_RULES:
        if pattern.search(description):
            return category
    return _DEFAULT_CATEGORY


# ---------------------------------------------------------------------------
# Topological sort (Kahn's algorithm)
# ---------------------------------------------------------------------------

def _topological_sort(
    task_ids: List[str],
    edges: List[DependencyEdge],
) -> List[str]:
    """
    Return a topologically sorted list of task_ids.

    Raises:
        AgentExecutionError: if a cycle is detected in the dependency graph
    """
    in_degree: Dict[str, int] = {tid: 0 for tid in task_ids}
    adjacency: Dict[str, List[str]] = defaultdict(list)

    for edge in edges:
        if edge.source_task_id in in_degree and edge.target_task_id in in_degree:
            adjacency[edge.source_task_id].append(edge.target_task_id)
            in_degree[edge.target_task_id] += 1

    queue: deque[str] = deque(tid for tid, deg in in_degree.items() if deg == 0)
    order: List[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in adjacency[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(order) != len(task_ids):
        raise AgentExecutionError(
            "DecompositionAgent",
            "Cycle detected in dependency graph — topological sort failed",
        )

    return order


# ---------------------------------------------------------------------------
# DecompositionAgent
# ---------------------------------------------------------------------------

class DecompositionAgent(BaseAgent[SharedContext, DecompositionResult]):
    """
    Breaks a query into a dependency-ordered graph of typed subtasks.

    Strategy (heuristic / LLM-free for offline testability):
      1. Split query on conjunctions / punctuation to seed candidate subtasks.
      2. Classify each subtask by keyword matching.
      3. Build dependency edges: retrieval always precedes reasoning/synthesis.
      4. Topologically sort to produce execution_order.

    Override _decompose_query() to inject an LLM-backed decomposition strategy
    without changing any contract or orchestration logic.
    """

    TIMEOUT_SECONDS = 30.0

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def validate_input(self, context: SharedContext) -> None:
        super().validate_input(context)
        if not context.query.strip():
            raise AgentValidationError(self.agent_name, "SharedContext.query must not be empty")

    def validate_output(self, result: DecompositionResult) -> None:
        super().validate_output(result)
        if not result.subtasks:
            raise AgentValidationError(self.agent_name, "DecompositionResult must contain at least one subtask")

    # ------------------------------------------------------------------ #
    #  Core run                                                            #
    # ------------------------------------------------------------------ #

    def run(self, context: SharedContext) -> DecompositionResult:
        """
        Decompose context.query into subtasks with a dependency graph.

        Args:
            context: SharedContext with non-empty .query

        Returns:
            DecompositionResult

        Raises:
            AgentExecutionError: on decomposition or topo-sort failure
        """
        logger.info("Decomposing query | run_id=%s", context.run_id)

        try:
            raw_subtasks = self._decompose_query(context.query, context)
            subtasks     = self._build_subtasks(raw_subtasks)
            edges        = self._build_edges(subtasks)
            order        = _topological_sort([st.task_id for st in subtasks], edges)
        except AgentExecutionError:
            raise
        except Exception as exc:
            raise AgentExecutionError(self.agent_name, f"Decomposition failed: {exc}") from exc

        result = DecompositionResult(
            subtasks=subtasks,
            dependency_edges=edges,
            execution_order=order,
            metadata={
                "run_id":        context.run_id,
                "query_length":  len(context.query),
                "subtask_count": len(subtasks),
            },
        )

        logger.info(
            "Decomposition complete | subtasks=%d order=%s",
            len(subtasks),
            order,
        )
        context.store_agent_output(self.agent_name, result)
        return result

    # ------------------------------------------------------------------ #
    #  Override-friendly decomposition strategy                           #
    # ------------------------------------------------------------------ #

    def _decompose_query(
        self,
        query: str,
        context: SharedContext,
    ) -> List[str]:
        """
        Split query into raw string descriptions.

        Override this method to integrate an LLM call.

        Args:
            query   : the user's top-level question
            context : full SharedContext (conversation history, docs, etc.)

        Returns:
            list of raw description strings (non-empty)
        """
        max_subtasks: int = context.metadata.get("max_subtasks", 8)

        # Heuristic: split on '. ', '? ', '! ', ' and ', ' then ', ';'
        sentences = re.split(r"(?<=[.?!])\s+|;\s*|\band\b|\bthen\b", query)
        parts = [s.strip() for s in sentences if s.strip()]

        # Always ensure at least a "retrieve" and a "synthesize" pass
        if not parts:
            parts = [query]

        # Prepend implicit retrieval if not already present
        has_retrieval = any(_classify(p) == "retrieval" for p in parts)
        if not has_retrieval:
            parts.insert(0, f"Retrieve relevant information for: {query}")

        # Append implicit synthesis if not already present
        has_synthesis = any(_classify(p) == "synthesis" for p in parts)
        if not has_synthesis:
            parts.append(f"Synthesize findings into a final answer for: {query}")

        return parts[:max_subtasks]

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_subtasks(self, descriptions: List[str]) -> List[SubTask]:
        subtasks: List[SubTask] = []
        for i, desc in enumerate(descriptions):
            task_id  = f"task_{i:03d}_{uuid.uuid4().hex[:6]}"
            category = _classify(desc)
            subtasks.append(
                SubTask(
                    task_id=task_id,
                    description=desc,
                    status=TaskStatus.PENDING,
                    category=category,
                )
            )
        return subtasks

    def _build_edges(self, subtasks: List[SubTask]) -> List[DependencyEdge]:
        """
        Wire edges: retrieval tasks must precede reasoning/synthesis/critique.
        """
        edges: List[DependencyEdge] = []
        retrieval_ids = [st.task_id for st in subtasks if st.category == "retrieval"]
        downstream_categories = {"reasoning", "synthesis", "critique", "compression"}

        for st in subtasks:
            if st.category in downstream_categories:
                for ret_id in retrieval_ids:
                    if ret_id != st.task_id:
                        edges.append(
                            DependencyEdge(
                                source_task_id=ret_id,
                                target_task_id=st.task_id,
                                edge_type="depends_on",
                            )
                        )
                        st.dependencies.append(ret_id)

        return edges


# ---------------------------------------------------------------------------
# Debug entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        stream=sys.stdout,
    )

    print("=" * 60)
    print("DecompositionAgent — debug run")
    print("=" * 60)

    ctx = SharedContext(
        query=(
            "Find recent papers on transformer attention mechanisms, "
            "then compare their efficiency claims, "
            "and synthesize a summary with citations."
        ),
        metadata={"max_subtasks": 6},
    )
    ctx.add_message("user", ctx.query)

    agent  = DecompositionAgent()
    result = agent(ctx)

    print("\n--- Subtasks ---")
    for st in result.subtasks:
        print(f"  [{st.task_id}] ({st.category}) {st.description}")
        if st.dependencies:
            print(f"         depends on: {st.dependencies}")

    print("\n--- Dependency Edges ---")
    for edge in result.dependency_edges:
        print(f"  {edge.source_task_id} --{edge.edge_type}--> {edge.target_task_id}")

    print("\n--- Execution Order ---")
    print(" → ".join(result.execution_order))

    print("\n--- Metadata ---")
    print(json.dumps(result.metadata, indent=2, default=str))
    print("\nSchema validation: PASSED ✓")
