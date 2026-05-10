"""
Module: contracts/event_contracts.py
======================================
Purpose:
    Defines all Pydantic contracts for pipeline events emitted during
    execution. Events are immutable records used for observability,
    audit logs, and downstream consumers (e.g. Kafka, Datadog, OpenTelemetry).

    Events are NEVER mutated after creation — they are append-only records.

Input Datatypes:
    None (pure schema definitions)

Output Datatypes:
    - ExecutionEvent          (base event)
    - ToolExecutionEvent      (tool-specific event)
    - AgentExecutionEvent     (agent-specific event)
    - PolicyViolationEvent    (policy/safety-specific event)

Dependencies:
    - pydantic >= 2.0
    - Python 3.11+

SOLID Principle:
    Single Responsibility — this file ONLY defines event contracts.
    Open/Closed — new event types extend ExecutionEvent, never modify the base.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Top-level category of a pipeline event."""

    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"

    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"

    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    TOOL_RETRIED = "tool.retried"

    POLICY_VIOLATION = "policy.violation"
    POLICY_CLEARED = "policy.cleared"


class EventSeverity(str, Enum):
    """Severity level of the event for alerting purposes."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PolicyCategory(str, Enum):
    """Category of a policy violation."""

    PII = "pii"
    TOXICITY = "toxicity"
    HALLUCINATION = "hallucination"
    COPYRIGHT = "copyright"
    PROMPT_INJECTION = "prompt_injection"
    RATE_LIMIT = "rate_limit"
    DATA_LEAK = "data_leak"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Base Event
# ---------------------------------------------------------------------------


class ExecutionEvent(BaseModel):
    """
    Base class for all pipeline execution events.

    All events are immutable append-only records. Do not add mutable
    helpers to this class or its subclasses.

    Fields
    ------
    event_id : str
        UUID uniquely identifying this event.
    event_type : EventType
        Category of this event.
    severity : EventSeverity
        Alerting severity level.
    job_id : str
        Pipeline run that emitted this event.
    emitted_at : datetime
        UTC timestamp when this event was emitted.
    schema_version : str
        Version of the event schema (semver).
    payload : dict[str, Any]
        Arbitrary extra data specific to this event type.
    trace_id : str | None
        Optional distributed tracing ID (e.g. OpenTelemetry trace ID).
    span_id : str | None
        Optional distributed tracing span ID.
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="UUID uniquely identifying this event.",
    )
    event_type: EventType = Field(..., description="Event category.")
    severity: EventSeverity = Field(
        default=EventSeverity.INFO,
        description="Alerting severity level.",
    )
    job_id: str = Field(..., description="Pipeline run that emitted this event.")
    emitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when event was emitted.",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Event schema version (semver).",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra data for this event type.",
    )
    trace_id: str | None = Field(
        default=None,
        description="Distributed tracing trace ID (OpenTelemetry compatible).",
    )
    span_id: str | None = Field(
        default=None,
        description="Distributed tracing span ID.",
    )

    model_config = {"frozen": True}  # Events are immutable after creation


# ---------------------------------------------------------------------------
# ToolExecutionEvent
# ---------------------------------------------------------------------------


class ToolExecutionEvent(ExecutionEvent):
    """
    Event emitted at tool execution boundaries (start, success, failure, retry).

    Additional Fields
    -----------------
    tool_name : str
        Registered name of the tool.
    correlation_id : str
        Matching ToolRequest.correlation_id.
    agent_name : str
        Agent that triggered this tool call.
    latency_ms : float | None
        Wall-clock duration (None for start events).
    retries_used : int
        Retry count at time of event emission.
    error_message : str | None
        Set on FAILURE and RETRIED events.
    """

    tool_name: str = Field(..., description="Registered tool name.")
    correlation_id: str = Field(
        ..., description="Matching ToolRequest.correlation_id."
    )
    agent_name: str = Field(..., description="Agent that triggered this tool call.")
    latency_ms: float | None = Field(
        default=None,
        description="Execution duration in ms (None for start events).",
    )
    retries_used: int = Field(
        default=0,
        ge=0,
        description="Retry count at event emission.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error description for FAILURE/RETRIED events.",
    )

    @model_validator(mode="after")
    def validate_tool_event_type(self) -> "ToolExecutionEvent":
        allowed = {
            EventType.TOOL_STARTED,
            EventType.TOOL_COMPLETED,
            EventType.TOOL_FAILED,
            EventType.TOOL_RETRIED,
        }
        if self.event_type not in allowed:
            raise ValueError(
                f"ToolExecutionEvent.event_type must be one of {allowed}; "
                f"got '{self.event_type}'."
            )
        return self


# ---------------------------------------------------------------------------
# AgentExecutionEvent
# ---------------------------------------------------------------------------


class AgentExecutionEvent(ExecutionEvent):
    """
    Event emitted at agent execution boundaries (start, success, failure).

    Additional Fields
    -----------------
    agent_name : str
        Name of the agent involved.
    agent_class : str
        Python class name of the agent.
    latency_ms : float | None
        Wall-clock duration (None for start events).
    confidence_score : float | None
        Confidence from the agent's output (None for start events).
    token_count : int
        Total tokens consumed by this agent invocation.
    error_message : str | None
        Set on FAILURE events.
    """

    agent_name: str = Field(..., description="Name of the agent.")
    agent_class: str = Field(
        default="unknown",
        description="Python class name of the agent.",
    )
    latency_ms: float | None = Field(
        default=None,
        description="Execution duration in ms (None for start events).",
    )
    confidence_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Agent output confidence (None for start events).",
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed by this agent invocation.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error description for FAILURE events.",
    )

    @model_validator(mode="after")
    def validate_agent_event_type(self) -> "AgentExecutionEvent":
        allowed = {
            EventType.AGENT_STARTED,
            EventType.AGENT_COMPLETED,
            EventType.AGENT_FAILED,
        }
        if self.event_type not in allowed:
            raise ValueError(
                f"AgentExecutionEvent.event_type must be one of {allowed}; "
                f"got '{self.event_type}'."
            )
        return self


# ---------------------------------------------------------------------------
# PolicyViolationEvent
# ---------------------------------------------------------------------------


class PolicyViolationEvent(ExecutionEvent):
    """
    Event emitted when a policy violation is detected anywhere in the pipeline.

    Additional Fields
    -----------------
    violation_id : str
        Unique ID for this specific violation instance.
    category : PolicyCategory
        Type of policy violated.
    description : str
        Human-readable description of the violation.
    agent_name : str
        Agent that triggered or detected the violation.
    affected_content_snippet : str | None
        Short (truncated) snippet of the offending content for debugging.
        Must be ≤ 200 characters.
    auto_remediated : bool
        Whether the pipeline automatically mitigated this violation.
    remediation_action : str | None
        Description of the auto-remediation performed, if any.
    """

    violation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for this violation instance.",
    )
    category: PolicyCategory = Field(
        ..., description="Type of policy violated."
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Human-readable violation description.",
    )
    agent_name: str = Field(
        ..., description="Agent that triggered or detected the violation."
    )
    affected_content_snippet: str | None = Field(
        default=None,
        max_length=200,
        description="Truncated snippet of offending content (≤200 chars).",
    )
    auto_remediated: bool = Field(
        default=False,
        description="True if the pipeline automatically mitigated this violation.",
    )
    remediation_action: str | None = Field(
        default=None,
        description="Description of auto-remediation performed.",
    )

    @model_validator(mode="after")
    def validate_policy_event_type(self) -> "PolicyViolationEvent":
        allowed = {EventType.POLICY_VIOLATION, EventType.POLICY_CLEARED}
        if self.event_type not in allowed:
            raise ValueError(
                f"PolicyViolationEvent.event_type must be one of {allowed}; "
                f"got '{self.event_type}'."
            )
        return self

    @model_validator(mode="after")
    def remediation_requires_action(self) -> "PolicyViolationEvent":
        if self.auto_remediated and not self.remediation_action:
            raise ValueError(
                "remediation_action must be set when auto_remediated is True."
            )
        return self


# ---------------------------------------------------------------------------
# Standalone debug entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("event_contracts.py — standalone debug mode")
    print("=" * 60)

    # ExecutionEvent (base)
    base_evt = ExecutionEvent(
        event_type=EventType.PIPELINE_STARTED,
        job_id="job-abc",
        severity=EventSeverity.INFO,
        payload={"query": "capital of France"},
    )
    print("\n[ExecutionEvent]", base_evt.event_type, base_evt.event_id[:8])

    # Immutability check
    print("\n[ExecutionEvent is frozen — mutation should raise]")
    try:
        base_evt.job_id = "hacked"
    except Exception as e:
        print(f"  Caught expected error: {type(e).__name__}")

    # ToolExecutionEvent
    tool_evt = ToolExecutionEvent(
        event_type=EventType.TOOL_COMPLETED,
        job_id="job-abc",
        tool_name="web_search",
        correlation_id="corr-xyz",
        agent_name="retrieval_agent",
        latency_ms=312.5,
        retries_used=1,
    )
    print("\n[ToolExecutionEvent]", tool_evt.tool_name, tool_evt.latency_ms)

    # AgentExecutionEvent
    agent_evt = AgentExecutionEvent(
        event_type=EventType.AGENT_COMPLETED,
        job_id="job-abc",
        agent_name="retrieval_agent",
        agent_class="RetrievalAgent",
        latency_ms=450.0,
        confidence_score=0.92,
        token_count=380,
    )
    print("\n[AgentExecutionEvent] confidence =", agent_evt.confidence_score)

    # PolicyViolationEvent
    pv_evt = PolicyViolationEvent(
        event_type=EventType.POLICY_VIOLATION,
        job_id="job-abc",
        severity=EventSeverity.WARNING,
        category=PolicyCategory.PII,
        description="Email address detected in query.",
        agent_name="policy_agent",
        affected_content_snippet="user@example.com",
        auto_remediated=True,
        remediation_action="Redacted email address from query.",
    )
    print("\n[PolicyViolationEvent] category =", pv_evt.category)

    # Validation: ToolExecutionEvent with wrong event_type
    print("\n[ToolExecutionEvent with wrong event_type should raise]")
    try:
        ToolExecutionEvent(
            event_type=EventType.PIPELINE_STARTED,
            job_id="x",
            tool_name="web_search",
            correlation_id="c",
            agent_name="a",
        )
    except Exception as e:
        print(f"  Caught expected error: {e}")

    print("\n✅ event_contracts.py debug complete.")