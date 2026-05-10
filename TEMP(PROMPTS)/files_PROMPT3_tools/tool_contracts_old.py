# contracts/tool_contracts.py (ORIGINAL)

"""
Module: contracts/tool_contracts.py
=====================================
Purpose:
    Defines all Pydantic contracts for tool invocation — requests, responses,
    failures, and retry metadata. Tools are discrete, stateless callables
    invoked by agents via the pipeline infrastructure.

Input Datatypes:
    None (pure schema definitions)

Output Datatypes:
    - ToolRequest
    - ToolResponse
    - ToolFailure
    - RetryMetadata

Dependencies:
    - pydantic >= 2.0
    - Python 3.11+

SOLID Principle:
    Single Responsibility — this file ONLY defines tool-level data contracts.
    Interface Segregation — ToolRequest and ToolResponse are decoupled types
    even though they share a correlation_id.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ToolStatus(str, Enum):
    """Terminal execution status of a tool call."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class RetryStrategy(str, Enum):
    """Strategy to apply between retry attempts."""

    NONE = "none"
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    JITTER = "jitter"


# ---------------------------------------------------------------------------
# RetryMetadata
# ---------------------------------------------------------------------------


class RetryMetadata(BaseModel):
    """
    Describes the retry policy and execution history for a tool call.

    Fields
    ------
    max_retries : int
        Maximum number of retry attempts allowed.
    retry_strategy : RetryStrategy
        Backoff strategy to use between attempts.
    base_delay_ms : float
        Base delay in milliseconds between retries.
    max_delay_ms : float
        Upper cap on delay regardless of backoff calculation.
    retries_used : int
        Number of retries actually consumed during execution.
    retry_timestamps : list[datetime]
        UTC timestamps of each retry attempt (excluding the first attempt).
    last_error : str | None
        Error message from the most recent failed attempt.
    """

    max_retries: int = Field(default=3, ge=0, description="Max retry attempts.")
    retry_strategy: RetryStrategy = Field(
        default=RetryStrategy.EXPONENTIAL,
        description="Backoff strategy between retries.",
    )
    base_delay_ms: float = Field(
        default=500.0,
        ge=0.0,
        description="Base delay between retries in ms.",
    )
    max_delay_ms: float = Field(
        default=10_000.0,
        ge=0.0,
        description="Maximum delay cap in ms.",
    )
    retries_used: int = Field(
        default=0,
        ge=0,
        description="Retries consumed during this execution.",
    )
    retry_timestamps: list[datetime] = Field(
        default_factory=list,
        description="UTC timestamps of each retry attempt.",
    )
    last_error: str | None = Field(
        default=None,
        description="Error message from the most recent failed attempt.",
    )

    @model_validator(mode="after")
    def retries_used_within_budget(self) -> "RetryMetadata":
        if self.retries_used > self.max_retries:
            raise ValueError(
                f"retries_used ({self.retries_used}) exceeds "
                f"max_retries ({self.max_retries})."
            )
        return self

    def next_delay_ms(self) -> float:
        """Calculate the delay before the next retry (does not mutate state)."""
        if self.retry_strategy == RetryStrategy.NONE:
            return 0.0
        if self.retry_strategy == RetryStrategy.FIXED:
            delay = self.base_delay_ms
        else:
            # exponential (with optional jitter handled externally)
            delay = self.base_delay_ms * (2 ** self.retries_used)
        return min(delay, self.max_delay_ms)


# ---------------------------------------------------------------------------
# ToolRequest
# ---------------------------------------------------------------------------


class ToolRequest(BaseModel):
    """
    A validated request to invoke a specific tool.

    Fields
    ------
    tool_name : str
        Registered name of the tool to invoke.
    correlation_id : str
        Unique ID linking this request to its ToolResponse.
    job_id : str
        Pipeline run identifier for traceability.
    agent_name : str
        Name of the agent issuing this request.
    parameters : dict[str, Any]
        Tool-specific input parameters (validated by the tool itself).
    timeout_ms : float
        Maximum allowed execution time before declaring TIMEOUT.
    retry_metadata : RetryMetadata
        Retry policy for this request.
    requested_at : datetime
        UTC timestamp when the request was created.
    idempotency_key : str | None
        Optional key to allow safe retries without double execution.
    """

    tool_name: str = Field(..., min_length=1, description="Registered tool name.")
    correlation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Links request to its matching ToolResponse.",
    )
    job_id: str = Field(..., description="Pipeline run identifier.")
    agent_name: str = Field(..., description="Requesting agent name.")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific input parameters.",
    )
    timeout_ms: float = Field(
        default=30_000.0,
        gt=0.0,
        description="Execution timeout in milliseconds.",
    )
    retry_metadata: RetryMetadata = Field(
        default_factory=RetryMetadata,
        description="Retry policy for this invocation.",
    )
    requested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when request was created.",
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Optional idempotency key to prevent double execution on retry.",
    )

    @field_validator("tool_name")
    @classmethod
    def tool_name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("tool_name must not be blank.")
        return v.strip()


# ---------------------------------------------------------------------------
# ToolResponse
# ---------------------------------------------------------------------------


class ToolResponse(BaseModel):
    """
    Validated response from a tool execution.

    Fields
    ------
    tool_name : str
        Name of the tool that produced this response.
    correlation_id : str
        Must match the originating ToolRequest.correlation_id.
    job_id : str
        Pipeline run identifier.
    status : ToolStatus
        Terminal status of this execution.
    stdout : str
        Standard output captured from the tool.
    stderr : str
        Standard error captured from the tool.
    result : Any
        Parsed tool result payload (tool-specific structure).
    latency_ms : float
        Wall-clock execution duration in milliseconds.
    success : bool
        Derived convenience flag — True iff status == SUCCESS.
    retries_used : int
        Number of retries consumed (mirrors RetryMetadata.retries_used).
    started_at : datetime
        UTC timestamp when tool execution began.
    completed_at : datetime
        UTC timestamp when tool execution finished.
    error_message : str | None
        Human-readable error description on failure.
    metadata : dict[str, Any]
        Arbitrary extra metadata (e.g. cache hit, tool version).
    """

    tool_name: str = Field(..., description="Name of the tool that ran.")
    correlation_id: str = Field(
        ...,
        description="Matches the originating ToolRequest.correlation_id.",
    )
    job_id: str = Field(..., description="Pipeline run identifier.")
    status: ToolStatus = Field(default=ToolStatus.SUCCESS)
    stdout: str = Field(default="", description="Standard output from the tool.")
    stderr: str = Field(default="", description="Standard error from the tool.")
    result: Any = Field(
        default=None,
        description="Parsed tool result payload (tool-specific).",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock execution duration in ms.",
    )
    success: bool = Field(
        default=True,
        description="True iff status == SUCCESS (derived).",
    )
    retries_used: int = Field(
        default=0,
        ge=0,
        description="Retry attempts consumed.",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC start timestamp.",
    )
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC completion timestamp.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error description if status is not SUCCESS.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra metadata.",
    )

    @model_validator(mode="after")
    def derive_success_flag(self) -> "ToolResponse":
        self.success = self.status == ToolStatus.SUCCESS
        return self

    @model_validator(mode="after")
    def failure_requires_error(self) -> "ToolResponse":
        if self.status in (ToolStatus.FAILURE, ToolStatus.TIMEOUT) and not self.error_message:
            raise ValueError(
                "error_message must be set when status is FAILURE or TIMEOUT."
            )
        return self


# ---------------------------------------------------------------------------
# ToolFailure
# ---------------------------------------------------------------------------


class ToolFailure(BaseModel):
    """
    Structured failure record created when a tool exhausts all retries
    or encounters an unrecoverable error.

    Fields
    ------
    tool_name : str
        Tool that failed.
    correlation_id : str
        Matches the originating ToolRequest.
    job_id : str
        Pipeline run identifier.
    error_type : str
        Short error category (e.g. "timeout", "auth_error", "parse_error").
    error_message : str
        Detailed human-readable error description.
    traceback : str | None
        Optional full Python traceback string.
    retry_metadata : RetryMetadata
        State of retries at time of final failure.
    failed_at : datetime
        UTC timestamp of the final failure.
    is_retryable : bool
        Whether a higher-level orchestrator may safely retry the whole tool call.
    """

    tool_name: str = Field(..., description="Tool that failed.")
    correlation_id: str = Field(..., description="Matching ToolRequest correlation_id.")
    job_id: str = Field(..., description="Pipeline run identifier.")
    error_type: str = Field(
        ...,
        description="Short error category (timeout / auth_error / parse_error / etc.).",
    )
    error_message: str = Field(..., description="Detailed error description.")
    traceback: str | None = Field(
        default=None,
        description="Full Python traceback string (optional).",
    )
    retry_metadata: RetryMetadata = Field(
        ...,
        description="Retry state at time of final failure.",
    )
    failed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of final failure.",
    )
    is_retryable: bool = Field(
        default=False,
        description="Whether an orchestrator may safely retry this tool call.",
    )


# ---------------------------------------------------------------------------
# Standalone debug entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from datetime import timedelta

    print("=" * 60)
    print("tool_contracts.py — standalone debug mode")
    print("=" * 60)

    # RetryMetadata
    rm = RetryMetadata(max_retries=3, base_delay_ms=200.0, retries_used=1)
    print(f"\n[RetryMetadata] next_delay_ms = {rm.next_delay_ms():.1f}")

    # ToolRequest
    req = ToolRequest(
        tool_name="web_search",
        job_id="job-abc",
        agent_name="retrieval_agent",
        parameters={"query": "capital of France", "top_k": 5},
        timeout_ms=5000.0,
    )
    print("\n[ToolRequest]", req.tool_name, req.correlation_id[:8])

    # ToolResponse — success
    now = datetime.now(timezone.utc)
    resp = ToolResponse(
        tool_name="web_search",
        correlation_id=req.correlation_id,
        job_id="job-abc",
        status=ToolStatus.SUCCESS,
        stdout='{"answer": "Paris"}',
        stderr="",
        result={"answer": "Paris"},
        latency_ms=342.0,
        retries_used=0,
        started_at=now,
        completed_at=now + timedelta(milliseconds=342),
    )
    print("\n[ToolResponse] success =", resp.success)

    # ToolResponse — failure validation
    print("\n[ToolResponse FAILURE without error_message should raise]")
    try:
        ToolResponse(
            tool_name="web_search",
            correlation_id="x",
            job_id="job-abc",
            status=ToolStatus.FAILURE,
        )
    except Exception as e:
        print(f"  Caught expected error: {e}")

    # ToolFailure
    tf = ToolFailure(
        tool_name="web_search",
        correlation_id=req.correlation_id,
        job_id="job-abc",
        error_type="timeout",
        error_message="Tool exceeded 5000ms timeout after 3 retries.",
        retry_metadata=RetryMetadata(max_retries=3, retries_used=3),
        is_retryable=False,
    )
    print("\n[ToolFailure] error_type =", tf.error_type)

    # RetryMetadata overbudget validation
    print("\n[RetryMetadata retries_used > max_retries should raise]")
    try:
        RetryMetadata(max_retries=2, retries_used=5)
    except Exception as e:
        print(f"  Caught expected error: {e}")

    print("\n✅ tool_contracts.py debug complete.")