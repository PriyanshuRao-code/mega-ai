"""
api/models.py
=============
Purpose   : All Pydantic request / response contracts used across the API layer.
Imports   : pydantic, typing, datetime, uuid, enum
Outputs   : Pydantic BaseModel subclasses (no side-effects)
Dependencies: None (pure data-layer)
Exceptions: ValidationError raised automatically by Pydantic on bad input
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class RewriteDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


class SSEEventType(str, Enum):
    ACTIVE_AGENT = "active_agent"
    ACTIVE_TOOL = "active_tool"
    TOKEN_STREAM = "token_stream"
    CONTEXT_BUDGET = "context_budget"
    ERROR = "error"
    DONE = "done"


# ---------------------------------------------------------------------------
# 1. Submit Query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Input for POST /query"""
    query: str = Field(..., min_length=1, max_length=32_768, description="User query text")
    session_id: Optional[str] = Field(default=None, description="Existing session to resume")
    config_overrides: Optional[Dict[str, Any]] = Field(default=None, description="Runtime config patches")
    stream: bool = Field(default=False, description="If true, response is streamed via SSE")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Summarise the Q3 earnings report",
                "session_id": None,
                "config_overrides": {"max_tokens": 2048},
                "stream": True,
            }
        }


class QueryResponse(BaseModel):
    """Output for POST /query (non-streaming)"""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    status: AgentStatus
    answer: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 2. Execution Trace
# ---------------------------------------------------------------------------

class TraceStep(BaseModel):
    step_index: int
    agent_name: str
    tool_name: Optional[str] = None
    input_summary: str
    output_summary: str
    latency_ms: int
    tokens_used: int
    timestamp: datetime


class ExecutionTraceResponse(BaseModel):
    """Output for GET /runs/{run_id}/trace"""
    run_id: str
    session_id: str
    status: AgentStatus
    steps: List[TraceStep]
    total_tokens: int
    total_latency_ms: int


# ---------------------------------------------------------------------------
# 3. Eval Summary
# ---------------------------------------------------------------------------

class EvalMetric(BaseModel):
    name: str
    score: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0)
    details: Optional[str] = None


class EvalSummaryResponse(BaseModel):
    """Output for GET /runs/{run_id}/eval"""
    run_id: str
    evaluated_at: datetime
    overall_score: float = Field(..., ge=0.0, le=1.0)
    metrics: List[EvalMetric]
    passed: bool
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# 4. Approve / Reject Rewrite
# ---------------------------------------------------------------------------

class RewriteDecisionRequest(BaseModel):
    """Input for POST /runs/{run_id}/rewrite"""
    decision: RewriteDecision
    reviewer_id: str = Field(..., min_length=1)
    feedback: Optional[str] = Field(default=None, max_length=4096)

    class Config:
        json_schema_extra = {
            "example": {
                "decision": "approve",
                "reviewer_id": "user-42",
                "feedback": "Looks good",
            }
        }


class RewriteDecisionResponse(BaseModel):
    """Output for POST /runs/{run_id}/rewrite"""
    run_id: str
    decision: RewriteDecision
    reviewer_id: str
    decided_at: datetime = Field(default_factory=datetime.utcnow)
    next_run_id: Optional[str] = Field(
        default=None,
        description="New run_id if rewrite triggers a new execution",
    )


# ---------------------------------------------------------------------------
# 5. Targeted Re-evaluation
# ---------------------------------------------------------------------------

class ReEvalRequest(BaseModel):
    """Input for POST /runs/{run_id}/reeval"""
    metric_names: List[str] = Field(..., min_length=1, description="Metrics to re-run")
    reason: Optional[str] = Field(default=None, max_length=1024)

    class Config:
        json_schema_extra = {
            "example": {
                "metric_names": ["faithfulness", "groundedness"],
                "reason": "Updated rubric after human review",
            }
        }


class ReEvalResponse(BaseModel):
    """Output for POST /runs/{run_id}/reeval"""
    run_id: str
    reeval_job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    queued_at: datetime = Field(default_factory=datetime.utcnow)
    metric_names: List[str]
    status: AgentStatus = AgentStatus.PENDING


# ---------------------------------------------------------------------------
# SSE event envelope
# ---------------------------------------------------------------------------

class SSEEvent(BaseModel):
    """Serialised payload for every SSE message."""
    event: SSEEventType
    run_id: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Generic error envelope
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str
    message: str
    field: Optional[str] = None


class ErrorResponse(BaseModel):
    status: int
    errors: List[ErrorDetail]
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
