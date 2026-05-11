# contracts/models.py (along with orchestrator)
"""
contracts/models.py
────────────────────────────────────────────────────────────────────────────────
Immutable data contracts for the multi-agent orchestration system.

DO NOT MODIFY — consumed by orchestrator layer and all agents.

Imports  : dataclasses, datetime, typing, enum, uuid
Exports  : SharedContext, ToolResponse, AgentExecutionEvent, ExecutionEvent,
           EventType, ExecutionStatus, PolicyViolation
Exceptions: (none raised here — pure data definitions)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


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

@dataclass
class ToolResponse:
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
    metadata   : Dict[str, Any]        = field(default_factory=dict)


@dataclass
class PolicyViolation:
    """Structured record of a policy or safety breach detected during execution."""
    rule        : str
    severity    : str                  # "low" | "medium" | "high" | "critical"
    agent_name  : str
    description : str
    timestamp   : datetime             = field(default_factory=datetime.utcnow)


@dataclass
class AgentExecutionEvent:
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
    timestamp  : datetime              = field(default_factory=datetime.utcnow)
    metadata   : Dict[str, Any]        = field(default_factory=dict)
    task_id    : str                   = field(default_factory=lambda: str(uuid.uuid4()))
    event_id   : str                   = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ExecutionEvent:
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
    finished_at      : UTC end time (None if still running).
    error            : Top-level error message if the run failed.
    """
    task_id           : str
    status            : ExecutionStatus
    agent_events      : List[AgentExecutionEvent]  = field(default_factory=list)
    policy_violations : List[PolicyViolation]      = field(default_factory=list)
    total_tokens_used : int                        = 0
    started_at        : datetime                   = field(default_factory=datetime.utcnow)
    finished_at       : Optional[datetime]         = None
    error             : Optional[str]              = None


@dataclass
class SharedContext:
    """
    Mutable state shared across all agents within a single orchestration run.

    Fields
    ------
    task_id          : Unique run identifier.
    goal             : Human-readable description of the top-level task.
    messages         : Conversation / instruction history.
    agent_outputs    : Accumulated ToolResponse results keyed by agent_name.
    dependency_graph : Adjacency list — agent → list of agents it depends on.
    agent_statuses   : Current ExecutionStatus per agent name.
    token_budget     : Maximum allowed token spend across all agents.
    tokens_used      : Running total of tokens consumed so far.
    metadata         : Arbitrary extra payload (user_id, session, …).
    policy_flags     : Accumulated PolicyViolation records.
    available_agents : Names of all agents registered with the orchestrator.
    completed_agents : Set of agents that have finished successfully.
    """
    task_id          : str
    goal             : str
    messages         : List[Dict[str, str]]             = field(default_factory=list)
    agent_outputs    : Dict[str, ToolResponse]          = field(default_factory=dict)
    dependency_graph : Dict[str, List[str]]             = field(default_factory=dict)
    agent_statuses   : Dict[str, ExecutionStatus]       = field(default_factory=dict)
    token_budget     : int                              = 100_000
    tokens_used      : int                              = 0
    metadata         : Dict[str, Any]                  = field(default_factory=dict)
    policy_flags     : List[PolicyViolation]            = field(default_factory=list)
    available_agents : List[str]                        = field(default_factory=list)
    completed_agents : List[str]                        = field(default_factory=list)

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def remaining_budget(self) -> int:
        return self.token_budget - self.tokens_used

    @property
    def budget_exhausted(self) -> bool:
        return self.tokens_used >= self.token_budget

    def record_agent_output(self, response: ToolResponse) -> None:
        self.agent_outputs[response.agent_name] = response
        self.tokens_used += response.tokens_used
        if response.success:
            if response.agent_name not in self.completed_agents:
                self.completed_agents.append(response.agent_name)
            self.agent_statuses[response.agent_name] = ExecutionStatus.COMPLETED
        else:
            self.agent_statuses[response.agent_name] = ExecutionStatus.FAILED