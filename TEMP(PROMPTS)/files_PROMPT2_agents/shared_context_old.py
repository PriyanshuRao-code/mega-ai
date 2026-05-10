# contracts/shared_context.py (ORIGINAL)

"""
Module: contracts/shared_context.py
=====================================
Purpose:
    Defines the SharedContext class — the SINGLE source of truth passed between
    all agents in the multi-agent pipeline. Agents do NOT call each other directly;
    they read from and write to SharedContext only.

Input Datatypes:
    None (constructed externally before pipeline execution begins)

Output Datatypes:
    SharedContext — a fully validated Pydantic model

Dependencies:
    - pydantic >= 2.0
    - Python 3.11+

SOLID Principle:
    Single Responsibility — this file ONLY defines shared pipeline state.
    No business logic. No agent logic. No tool logic.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Supporting sub-models
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Tracks token consumption across the pipeline."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    def add(self, prompt: int, completion: int) -> None:
        """Accumulate token counts in-place."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion


class ExecutionTraceEntry(BaseModel):
    """Single entry in the execution trace log."""

    agent_name: str
    step: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Core SharedContext
# ---------------------------------------------------------------------------


class SharedContext(BaseModel):
    """
    Central shared state object passed through the entire multi-agent pipeline.

    All agents read inputs from and write outputs to this object.
    No direct agent-to-agent communication is permitted.

    Fields
    ------
    job_id : str
        Unique identifier for this pipeline execution run.
    query : str
        The original user query or task description.
    current_agent : str
        Name of the agent currently executing (set before each agent call).
    conversation_history : list[str]
        Ordered list of conversation turns (role: content strings).
    agent_outputs : dict[str, Any]
        Keyed by agent name. Each value is that agent's BaseAgentOutput dict.
    tool_outputs : list[Any]
        Ordered list of ToolResponse objects (serialised as dicts).
    citations : list[str]
        Collected citations/sources accumulated across all agents.
    token_usage : TokenUsage
        Aggregated token usage across all LLM calls.
    policy_violations : list[str]
        Human-readable policy violation messages raised during execution.
    execution_trace : list[ExecutionTraceEntry]
        Ordered log of every agent step for observability and debugging.
    metadata : dict[str, Any]
        Arbitrary key-value store for pipeline-wide configuration or flags.
    """

    # Identity
    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique pipeline run identifier.",
    )
    query: str = Field(..., min_length=1, description="Original user query.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Pipeline creation timestamp (UTC).",
    )

    # Routing
    current_agent: str = Field(
        default="",
        description="Name of the agent currently holding execution.",
    )

    # Conversation
    conversation_history: list[str] = Field(
        default_factory=list,
        description="Ordered list of conversation turns as 'role: content' strings.",
    )

    # Agent outputs
    agent_outputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyed by agent name; stores each agent's serialised output.",
    )

    # Tool outputs
    tool_outputs: list[Any] = Field(
        default_factory=list,
        description="Ordered list of serialised ToolResponse objects.",
    )

    # Citations
    citations: list[str] = Field(
        default_factory=list,
        description="Accumulated citation strings from all agents.",
    )

    # Token tracking
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Aggregated token usage across all LLM calls.",
    )

    # Policy
    policy_violations: list[str] = Field(
        default_factory=list,
        description="Human-readable descriptions of any detected policy violations.",
    )

    # Observability
    execution_trace: list[ExecutionTraceEntry] = Field(
        default_factory=list,
        description="Ordered execution log for each agent step.",
    )

    # Extensibility
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary pipeline-wide configuration or runtime flags.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank or whitespace-only.")
        return v.strip()

    # ------------------------------------------------------------------
    # Convenience helpers (no business logic — purely state management)
    # ------------------------------------------------------------------

    def set_current_agent(self, agent_name: str) -> None:
        """Update which agent is currently executing."""
        self.current_agent = agent_name

    def record_agent_output(self, agent_name: str, output: Any) -> None:
        """Store a serialised agent output keyed by agent name."""
        self.agent_outputs[agent_name] = output

    def append_tool_output(self, tool_output: Any) -> None:
        """Append a serialised ToolResponse to the tool outputs list."""
        self.tool_outputs.append(tool_output)

    def add_citation(self, citation: str) -> None:
        """Add a citation string if not already present."""
        if citation not in self.citations:
            self.citations.append(citation)

    def flag_policy_violation(self, message: str) -> None:
        """Record a policy violation message."""
        self.policy_violations.append(message)

    def trace(
        self,
        agent_name: str,
        step: str,
        duration_ms: float | None = None,
        notes: str | None = None,
    ) -> None:
        """Append an entry to the execution trace."""
        self.execution_trace.append(
            ExecutionTraceEntry(
                agent_name=agent_name,
                step=step,
                duration_ms=duration_ms,
                notes=notes,
            )
        )

    def has_violations(self) -> bool:
        """Return True if any policy violations have been recorded."""
        return len(self.policy_violations) > 0

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary dict for logging/debugging."""
        return {
            "job_id": self.job_id,
            "query": self.query[:80] + ("..." if len(self.query) > 80 else ""),
            "current_agent": self.current_agent,
            "agents_completed": list(self.agent_outputs.keys()),
            "tool_calls": len(self.tool_outputs),
            "citations": len(self.citations),
            "policy_violations": len(self.policy_violations),
            "total_tokens": self.token_usage.total_tokens,
            "trace_steps": len(self.execution_trace),
        }


# ---------------------------------------------------------------------------
# Standalone debug entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("SharedContext — standalone debug mode")
    print("=" * 60)

    ctx = SharedContext(query="What is the capital of France?")
    ctx.set_current_agent("retrieval_agent")
    ctx.trace("retrieval_agent", "start")
    ctx.record_agent_output("retrieval_agent", {"answer": "Paris", "confidence": 0.99})
    ctx.add_citation("https://en.wikipedia.org/wiki/France")
    ctx.token_usage.add(prompt=120, completion=30)
    ctx.trace("retrieval_agent", "end", duration_ms=342.5)
    ctx.flag_policy_violation("Example: PII detected in query (demo only)")

    print("\n[SharedContext summary]")
    print(json.dumps(ctx.summary(), indent=2))

    print("\n[Full model JSON]")
    print(ctx.model_dump_json(indent=2))

    print("\n[Validation — blank query should raise]")
    try:
        bad = SharedContext(query="   ")
    except Exception as e:
        print(f"  Caught expected error: {e}")

    print("\n✅ SharedContext debug complete.")