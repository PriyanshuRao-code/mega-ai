# contracts/agent_contracts.py (ORIGINAL)
"""
Module: contracts/agent_contracts.py
======================================
Purpose:
    Defines all Pydantic contracts for agent inputs and outputs.
    Every agent in the pipeline must accept a SharedContext and return
    a subclass of BaseAgentOutput. No business logic lives here.

Input Datatypes:
    None (these are pure schema definitions)

Output Datatypes:
    - BaseAgentInput
    - BaseAgentOutput
    - RetrievalResult
    - CritiqueResult
    - SynthesisResult
    - CompressionResult
    - DecompositionResult

Dependencies:
    - pydantic >= 2.0
    - Python 3.11+

SOLID Principle:
    Single Responsibility — this file ONLY defines agent-level data contracts.
    Open/Closed — new agent result types extend BaseAgentOutput without
    modifying existing contracts.
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


class AgentStatus(str, Enum):
    """Terminal status of an agent execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class ConfidenceTier(str, Enum):
    """Human-readable confidence band derived from a float score."""

    HIGH = "high"       # >= 0.80
    MEDIUM = "medium"   # >= 0.50
    LOW = "low"         # < 0.50

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceTier":
        if score >= 0.80:
            return cls.HIGH
        if score >= 0.50:
            return cls.MEDIUM
        return cls.LOW


# ---------------------------------------------------------------------------
# Provenance metadata (reused across all result types)
# ---------------------------------------------------------------------------


class ProvenanceMetadata(BaseModel):
    """
    Tracks where a result came from and how it was produced.

    Fields
    ------
    agent_name : str
        The name of the agent that produced this result.
    model_id : str
        The LLM model identifier used (e.g. "claude-sonnet-4-20250514").
    prompt_version : str
        Version tag of the prompt template used.
    run_id : str
        Unique ID for this specific agent invocation.
    source_references : list[str]
        Any upstream document IDs, URLs, or keys consulted.
    extra : dict[str, Any]
        Arbitrary extra provenance metadata.
    """

    agent_name: str = Field(..., description="Agent that produced this result.")
    model_id: str = Field(default="unknown", description="LLM model identifier.")
    prompt_version: str = Field(default="v0", description="Prompt template version.")
    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for this agent invocation.",
    )
    source_references: list[str] = Field(
        default_factory=list,
        description="Upstream document IDs, URLs, or keys consulted.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra provenance metadata.",
    )


# ---------------------------------------------------------------------------
# Base contracts
# ---------------------------------------------------------------------------


class BaseAgentInput(BaseModel):
    """
    Minimal validated input wrapper consumed by every agent.

    Agents receive a SharedContext, not this model directly, but
    BaseAgentInput may be used to validate fields extracted from context
    before agent processing begins.

    Fields
    ------
    job_id : str        — pipeline run identifier (from SharedContext)
    agent_name : str    — name of the consuming agent
    query : str         — the task/query this agent should handle
    extra : dict        — additional agent-specific parameters
    """

    job_id: str = Field(..., description="Pipeline run identifier.")
    agent_name: str = Field(..., description="Name of the consuming agent.")
    query: str = Field(..., min_length=1, description="Task or query for this agent.")
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific extra parameters.",
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank.")
        return v.strip()


class BaseAgentOutput(BaseModel):
    """
    Base class for all agent outputs.

    Every concrete agent output type inherits from this.

    Fields
    ------
    job_id : str                 — matches the pipeline run
    agent_name : str             — which agent produced this
    status : AgentStatus         — execution outcome
    confidence_score : float     — 0.0–1.0
    confidence_tier : ConfidenceTier  — derived HIGH/MEDIUM/LOW band
    provenance : ProvenanceMetadata
    started_at : datetime        — UTC start timestamp
    completed_at : datetime      — UTC end timestamp
    latency_ms : float           — wall-clock duration
    error_message : str | None   — set on FAILURE
    warnings : list[str]         — non-fatal issues
    """

    job_id: str = Field(..., description="Matching pipeline run identifier.")
    agent_name: str = Field(..., description="Agent that produced this output.")
    status: AgentStatus = Field(default=AgentStatus.SUCCESS)
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Model confidence in output quality (0.0–1.0).",
    )
    confidence_tier: ConfidenceTier = Field(
        default=ConfidenceTier.HIGH,
        description="Human-readable confidence band.",
    )
    provenance: ProvenanceMetadata = Field(
        ..., description="Provenance and traceability metadata."
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when agent execution started.",
    )
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when agent execution completed.",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock execution duration in milliseconds.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error description if status is FAILURE.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issue descriptions.",
    )

    @model_validator(mode="after")
    def derive_confidence_tier(self) -> "BaseAgentOutput":
        self.confidence_tier = ConfidenceTier.from_score(self.confidence_score)
        return self

    @model_validator(mode="after")
    def failure_requires_error_message(self) -> "BaseAgentOutput":
        if self.status == AgentStatus.FAILURE and not self.error_message:
            raise ValueError("error_message must be set when status is FAILURE.")
        return self


# ---------------------------------------------------------------------------
# Specialised result contracts
# ---------------------------------------------------------------------------


class RetrievalResult(BaseAgentOutput):
    """
    Output from a Retrieval Agent.

    Additional Fields
    -----------------
    retrieved_chunks : list[str]
        Raw text chunks retrieved from a knowledge source.
    source_ids : list[str]
        Identifiers for the sources that were queried.
    retrieval_strategy : str
        Description of the strategy used (e.g. "dense", "sparse", "hybrid").
    total_candidates : int
        Number of candidate chunks considered before filtering.
    returned_count : int
        Number of chunks returned after ranking/filtering.
    """

    retrieved_chunks: list[str] = Field(
        default_factory=list,
        description="Text chunks retrieved from knowledge source.",
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="Identifiers for queried sources.",
    )
    retrieval_strategy: str = Field(
        default="hybrid",
        description="Retrieval strategy used (dense/sparse/hybrid/etc.).",
    )
    total_candidates: int = Field(
        default=0,
        ge=0,
        description="Candidate chunks considered before filtering.",
    )
    returned_count: int = Field(
        default=0,
        ge=0,
        description="Chunks returned after ranking/filtering.",
    )

    @model_validator(mode="after")
    def returned_cannot_exceed_candidates(self) -> "RetrievalResult":
        if self.returned_count > self.total_candidates:
            raise ValueError(
                f"returned_count ({self.returned_count}) cannot exceed "
                f"total_candidates ({self.total_candidates})."
            )
        return self


class CritiqueResult(BaseAgentOutput):
    """
    Output from a Critique / Evaluation Agent.

    Additional Fields
    -----------------
    critique_text : str
        Human-readable critique of the target content.
    issues_found : list[str]
        Specific issues or flaws identified.
    suggestions : list[str]
        Actionable improvement suggestions.
    severity : str
        Overall severity: "none" | "minor" | "moderate" | "severe".
    target_agent : str
        Name of the agent whose output is being critiqued.
    """

    critique_text: str = Field(
        default="",
        description="Human-readable critique of the target content.",
    )
    issues_found: list[str] = Field(
        default_factory=list,
        description="Specific issues or flaws identified.",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable improvement suggestions.",
    )
    severity: str = Field(
        default="none",
        pattern=r"^(none|minor|moderate|severe)$",
        description="Overall severity level.",
    )
    target_agent: str = Field(
        default="",
        description="Name of the agent whose output was critiqued.",
    )


class SynthesisResult(BaseAgentOutput):
    """
    Output from a Synthesis Agent that merges multiple inputs.

    Additional Fields
    -----------------
    synthesised_text : str
        The final synthesised output text.
    input_sources : list[str]
        Agent names or source IDs that contributed to the synthesis.
    synthesis_method : str
        Description of the merge strategy used.
    word_count : int
        Word count of the synthesised text.
    """

    synthesised_text: str = Field(
        default="",
        description="Final synthesised output text.",
    )
    input_sources: list[str] = Field(
        default_factory=list,
        description="Agent names or source IDs contributing to synthesis.",
    )
    synthesis_method: str = Field(
        default="concat",
        description="Merge strategy: concat / map-reduce / hierarchical / etc.",
    )
    word_count: int = Field(
        default=0,
        ge=0,
        description="Word count of synthesised_text.",
    )

    @model_validator(mode="after")
    def compute_word_count(self) -> "SynthesisResult":
        if self.synthesised_text:
            self.word_count = len(self.synthesised_text.split())
        return self


class CompressionResult(BaseAgentOutput):
    """
    Output from a Compression / Summarisation Agent.

    Additional Fields
    -----------------
    compressed_text : str
        The compressed/summarised output.
    original_token_count : int
        Token count of the original input text.
    compressed_token_count : int
        Token count of the compressed output.
    compression_ratio : float
        Ratio of compressed to original token count (0.0–1.0).
    """

    compressed_text: str = Field(
        default="",
        description="Compressed or summarised output text.",
    )
    original_token_count: int = Field(
        default=0,
        ge=0,
        description="Token count of input text before compression.",
    )
    compressed_token_count: int = Field(
        default=0,
        ge=0,
        description="Token count of compressed output.",
    )
    compression_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Compressed / original token ratio.",
    )

    @model_validator(mode="after")
    def compute_ratio(self) -> "CompressionResult":
        if self.original_token_count > 0:
            self.compression_ratio = round(
                self.compressed_token_count / self.original_token_count, 4
            )
        return self


class DecompositionResult(BaseAgentOutput):
    """
    Output from a Query / Task Decomposition Agent.

    Additional Fields
    -----------------
    sub_tasks : list[str]
        The decomposed sub-tasks or sub-queries.
    decomposition_strategy : str
        Strategy used: "sequential" | "parallel" | "hierarchical" | etc.
    dependency_graph : dict[str, list[str]]
        Maps each sub-task to the list of sub-tasks it depends on.
    """

    sub_tasks: list[str] = Field(
        default_factory=list,
        description="Decomposed sub-tasks or sub-queries.",
    )
    decomposition_strategy: str = Field(
        default="sequential",
        description="Strategy: sequential / parallel / hierarchical.",
    )
    dependency_graph: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Maps sub-task → list of sub-tasks it depends on.",
    )


# ---------------------------------------------------------------------------
# Standalone debug entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from datetime import timedelta

    print("=" * 60)
    print("agent_contracts.py — standalone debug mode")
    print("=" * 60)

    prov = ProvenanceMetadata(
        agent_name="retrieval_agent",
        model_id="claude-sonnet-4-20250514",
        prompt_version="v1.2",
        source_references=["doc_001", "doc_042"],
    )
    now = datetime.now(timezone.utc)

    # RetrievalResult
    r = RetrievalResult(
        job_id="job-123",
        agent_name="retrieval_agent",
        confidence_score=0.91,
        provenance=prov,
        started_at=now,
        completed_at=now + timedelta(milliseconds=210),
        latency_ms=210.0,
        retrieved_chunks=["Paris is the capital of France.", "France has 68M people."],
        source_ids=["wiki_france"],
        retrieval_strategy="hybrid",
        total_candidates=50,
        returned_count=2,
    )
    print("\n[RetrievalResult]")
    print(json.loads(r.model_dump_json(indent=2))["confidence_tier"])

    # CritiqueResult
    c = CritiqueResult(
        job_id="job-123",
        agent_name="critique_agent",
        confidence_score=0.75,
        provenance=ProvenanceMetadata(agent_name="critique_agent"),
        latency_ms=88.0,
        critique_text="The retrieval is accurate but could include more context.",
        issues_found=["Missing historical context"],
        suggestions=["Add a sentence about French history"],
        severity="minor",
        target_agent="retrieval_agent",
    )
    print("\n[CritiqueResult severity]", c.severity)

    # SynthesisResult
    s = SynthesisResult(
        job_id="job-123",
        agent_name="synthesis_agent",
        confidence_score=0.88,
        provenance=ProvenanceMetadata(agent_name="synthesis_agent"),
        synthesised_text="Paris is the capital of France, a nation of 68 million people.",
        input_sources=["retrieval_agent", "critique_agent"],
        synthesis_method="map-reduce",
    )
    print("\n[SynthesisResult word_count]", s.word_count)

    # CompressionResult
    comp = CompressionResult(
        job_id="job-123",
        agent_name="compression_agent",
        confidence_score=0.82,
        provenance=ProvenanceMetadata(agent_name="compression_agent"),
        compressed_text="Paris: capital of France (68M pop.)",
        original_token_count=200,
        compressed_token_count=12,
    )
    print("\n[CompressionResult ratio]", comp.compression_ratio)

    # DecompositionResult
    d = DecompositionResult(
        job_id="job-123",
        agent_name="decomposition_agent",
        confidence_score=0.95,
        provenance=ProvenanceMetadata(agent_name="decomposition_agent"),
        sub_tasks=["Retrieve facts about France", "Retrieve capital city info"],
        decomposition_strategy="parallel",
        dependency_graph={
            "Retrieve facts about France": [],
            "Retrieve capital city info": [],
        },
    )
    print("\n[DecompositionResult sub_tasks]", d.sub_tasks)

    # Failure validation
    print("\n[Failure without error_message should raise]")
    try:
        BaseAgentOutput(
            job_id="x",
            agent_name="x",
            status=AgentStatus.FAILURE,
            provenance=ProvenanceMetadata(agent_name="x"),
        )
    except Exception as e:
        print(f"  Caught expected error: {e}")

    print("\n✅ agent_contracts.py debug complete.")