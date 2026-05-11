# contracts/tool_contracts.py (GENERATED ALONG WITH tools)

"""
contracts/tool_contracts.py
===========================
Canonical request / response envelope for every tool.

Imports   : pydantic models, enum, typing, uuid, datetime
Inputs    : (constructed by callers)
Outputs   : ToolRequest, ToolResponse, ToolStatus, SearchResult,
            ExecutionResult, SQLResult, ReflectionResult
Exceptions: — (pure data; no logic that raises)
Dependencies: stdlib only
"""

from __future__ import annotations

import uuid
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ──────────────────────────────────────────────
#  Status enum
# ──────────────────────────────────────────────

class ToolStatus(str, Enum):
    SUCCESS       = "success"
    ERROR         = "error"
    TIMEOUT       = "timeout"
    INVALID_INPUT = "invalid_input"
    EMPTY         = "empty_response"
    RETRIED       = "retried"


# ──────────────────────────────────────────────
#  Request envelope
# ──────────────────────────────────────────────

class ToolRequest(BaseModel):
    """
    Uniform input envelope for all tools.

    Fields
    ------
    tool_name   : logical name of the target tool
    payload     : dict carrying tool-specific parameters
    request_id  : auto-generated UUID4 (overridable for tracing)
    timestamp   : UTC creation time
    timeout     : per-request override (seconds); None → tool default
    metadata    : free-form dict for agent-level annotations
    """
    tool_name : str
    payload   : dict[str, Any]
    request_id: str                  = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp : datetime             = Field(default_factory=lambda: datetime.now(timezone.utc))
    timeout   : float | None         = None
    metadata  : dict[str, Any]       = Field(default_factory=dict)


# ──────────────────────────────────────────────
#  Typed result payloads
# ──────────────────────────────────────────────

class SearchResult(BaseModel):
    """Single web-search hit with relevance scoring."""
    title    : str
    url      : str
    snippet  : str
    score    : float          # 0.0 – 1.0
    rank     : int
    metadata : dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Outcome of sandboxed Python execution."""
    stdout     : str
    stderr     : str
    return_code: int
    timed_out  : bool = False
    exec_ms    : float = 0.0


class SQLResult(BaseModel):
    """NL→SQL conversion + query execution result."""
    nl_query    : str
    generated_sql: str
    columns     : list[str]
    rows        : list[list[Any]]
    row_count   : int
    exec_ms     : float = 0.0


class ReflectionResult(BaseModel):
    """Self-reflection analysis over previous outputs."""
    contradictions   : list[dict[str, Any]]
    summary          : str
    confidence_score : float          # 0.0 – 1.0
    flagged_segments : list[str]


# ──────────────────────────────────────────────
#  Response envelope
# ──────────────────────────────────────────────

class ToolResponse(BaseModel):
    """
    Uniform output envelope returned by every tool.

    Fields
    ------
    request_id : echoes ToolRequest.request_id
    tool_name  : name of the tool that produced this response
    status     : ToolStatus value
    data       : typed result payload (or None on failure)
    error      : human-readable error message (or None on success)
    attempts   : how many execution attempts were made
    duration_ms: wall-clock time for the final successful attempt
    timestamp  : UTC time the response was produced
    """
    request_id : str
    tool_name  : str
    status     : ToolStatus
    data       : Any | None          = None
    error      : str | None          = None
    attempts   : int                 = 1
    duration_ms: float               = 0.0
    timestamp  : datetime            = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── convenience constructors ──────────────────────────────────────── #

    @classmethod
    def success(
        cls,
        *,
        request_id : str,
        tool_name  : str,
        data       : Any,
        attempts   : int   = 1,
        duration_ms: float = 0.0,
    ) -> "ToolResponse":
        return cls(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolStatus.SUCCESS,
            data=data,
            attempts=attempts,
            duration_ms=duration_ms,
        )

    @classmethod
    def failure(
        cls,
        *,
        request_id: str,
        tool_name : str,
        status    : ToolStatus = ToolStatus.ERROR,
        error     : str,
        attempts  : int = 1,
    ) -> "ToolResponse":
        return cls(
            request_id=request_id,
            tool_name=tool_name,
            status=status,
            error=error,
            attempts=attempts,
        )

    @property
    def ok(self) -> bool:
        return self.status == ToolStatus.SUCCESS
