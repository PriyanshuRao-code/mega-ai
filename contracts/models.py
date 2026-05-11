# contracts/models.py (along with orchestrator)
"""
contracts/models.py
────────────────────────────────────────────────────────────────────────────────
Immutable data contracts for the multi-agent orchestration system.

DO NOT MODIFY — consumed by orchestrator layer and all agents.

Imports  : pydantic, datetime, typing, enum, uuid
Exports  : ToolResponse, AgentExecutionEvent, ExecutionEvent,
           EventType, ExecutionStatus, PolicyViolation
Exceptions: (none raised here — pure data definitions)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    AGENT_STARTED      = "AGENT_STARTED"
    AGENT_COMPLETED    = "AGENT_COMPLETED"
    AGENT_FAILED       = "AGENT_FAILED"
    AGENT_RETRYING     = "AGENT_RETRYING"
    AGENT_TIMEOUT      = "AGENT_TIMEOUT"
    ROUTING_DECISION   = "ROUTING_DECISION"
    ROUTING_AMBIGUOUS  = "ROUTING_AMBIGUOUS"
    POLICY_VIOLATION   = "POLICY_VIOLATION"
    BUDGET_EXCEEDED    = "BUDGET_EXCEEDED"
    ORCHESTRATION_DONE = "ORCHESTRATION_DONE"
    DEPENDENCY_BLOCKED = "DEPENDENCY_BLOCKED"
    SCHEDULER_CYCLE    = "SCHEDULER_CYCLE"


class ExecutionStatus(str, Enum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"
    SKIPPED   = "SKIPPED"
    TIMEOUT   = "TIMEOUT"


# ──────────────────────────────────────────────────────────────────────────────
# Core contracts
# ──────────────────────────────────────────────────────────────────────────────

class ToolResponse(BaseModel):
    """
    Result produced by a single agent execution.

    Fields
    ------
    agent_name   : Identifier of the agent that produced this response.
    output       : Arbitrary structured output (dict, str, list, …).
    tokens_used  : Token count consumed during this execution.
    success      : Whether the agent completed without error.
    error        : Error message when success=False; None otherwise.
    metadata     : Optional extra info (latency, model version, …).
    """
    agent_name : str
    output     : Any
    tokens_used: int
    success    : bool
    error      : Optional[str]         = None
    metadata   : Dict[str, Any]        = Field(default_factory=dict)


class PolicyViolation(BaseModel):
    """Structured record of a policy or safety breach detected during execution."""
    rule        : str
    severity    : str                  # "low" | "medium" | "high" | "critical"
    agent_name  : str
    description : str
    timestamp   : datetime             = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentExecutionEvent(BaseModel):
    """
    Fine-grained event emitted at every state transition of an agent.

    Fields
    ------
    event_type  : One of EventType enum values.
    agent_name  : Agent involved; may be "orchestrator" for system events.
    timestamp   : UTC time of the event.
    metadata    : Arbitrary extra payload (routing reason, retry count, …).
    task_id     : Parent task this event belongs to.
    event_id    : Unique ID for this event record.
    """
    event_type : EventType
    agent_name : str
    timestamp  : datetime              = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata   : Dict[str, Any]        = Field(default_factory=dict)
    task_id    : str                   = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id   : str                   = Field(default_factory=lambda: str(uuid.uuid4()))


class ExecutionEvent(BaseModel):
    """
    Top-level log record returned by the orchestrator after a full run.

    Fields
    ------
    task_id          : Unique identifier for the entire orchestration run.
    status           : Final ExecutionStatus of the run.
    agent_events     : Ordered list of AgentExecutionEvents emitted during run.
    policy_violations: Any policy breaches captured.
    total_tokens_used: Aggregate token consumption across all agents.
    started_at       : UTC start time.
    finished_at      : Optional[datetime]
    error            : Top-level error message if the run failed.
    """
    task_id           : str
    status            : ExecutionStatus
    agent_events      : List[AgentExecutionEvent]  = Field(default_factory=list)
    policy_violations : List[PolicyViolation]      = Field(default_factory=list)
    total_tokens_used : int                        = 0
    started_at        : datetime                   = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at       : Optional[datetime]         = None
    error             : Optional[str]              = None


