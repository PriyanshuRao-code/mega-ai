# contracts/agent_contracts.py (GENERATED ALONG WITH agents)
"""
contracts/agent_contracts.py
=============================
Strongly-typed output contracts (result pydantic models) for every agent in the
multi-agent pipeline.  These are the ONLY permitted return types from agent
run() implementations.

SOLID Alignment:
  - (S) One result dataclass per agent, one responsibility each
  - (O) Add new optional fields freely; never remove existing ones
  - (L) All result types are safe to use wherever a generic result is expected
  - (I) Agents import only the result type they produce
  - (D) Agents depend on these abstractions, not on each other

Imports:
  - pydantic models  (stdlib)
  - typing       (stdlib)
  - enum         (stdlib)

Exceptions:
  - ContractValidationError: raised by __post_init__ when required invariants
                             are violated
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared exception
# ---------------------------------------------------------------------------

class ContractValidationError(ValueError):
    """Raised when a result dataclass fails its invariant checks."""


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

class ConfidenceLevel(Enum):
    HIGH   = auto()
    MEDIUM = auto()
    LOW    = auto()
    UNKNOWN = auto()


class TaskStatus(Enum):
    PENDING    = auto()
    IN_PROGRESS = auto()
    COMPLETE   = auto()
    BLOCKED    = auto()


# ---------------------------------------------------------------------------
# DecompositionResult
# ---------------------------------------------------------------------------

class SubTask(BaseModel):
    """
    A single decomposed unit of work.

    Fields:
        task_id      : unique identifier within the run
        description  : human-readable task description
        status       : current TaskStatus
        dependencies : list of task_ids this task depends on
        category     : semantic label (e.g. 'retrieval', 'reasoning', 'synthesis')
        metadata     : arbitrary extra info
    """
    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = Field(default_factory=list)
    category: str = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DependencyEdge(BaseModel):
    """Directed edge in the dependency graph."""
    source_task_id: str
    target_task_id: str
    edge_type: str = "depends_on"  # e.g. 'depends_on', 'blocks', 'informs'


class DecompositionResult(BaseModel):
    """
    Output contract for DecompositionAgent.

    Fields:
        subtasks         : ordered list of SubTask objects
        dependency_edges : edges forming the dependency graph
        execution_order  : topologically sorted list of task_ids
        metadata         : arbitrary extra info

    Raises:
        ContractValidationError: if subtasks is empty
    """
    subtasks: List[SubTask]
    dependency_edges: List[DependencyEdge] = Field(default_factory=list)
    execution_order: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_model(self):
        if not self.subtasks:
            raise ContractValidationError("DecompositionResult.subtasks must not be empty")


        return self
# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """
    A single retrieved passage with provenance metadata.

    Fields:
        chunk_id    : unique chunk identifier
        doc_id      : parent document identifier
        source      : origin URI / label
        content     : passage text
        score       : relevance score (0.0 – 1.0)
        hop         : retrieval hop index (0 = direct, 1+ = multi-hop)
        citations   : list of (label, uri) citation tuples
        metadata    : arbitrary extra info
    """
    chunk_id: str
    doc_id: str
    source: str
    content: str
    score: float = 0.0
    hop: int = 0
    citations: List[Tuple[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_model(self):
        if not (0.0 <= self.score <= 1.0):
            raise ContractValidationError(
                f"RetrievedChunk.score must be in [0, 1]; got {self.score}"
            )
        if self.hop < 0:
            raise ContractValidationError(
                f"RetrievedChunk.hop must be >= 0; got {self.hop}"
            )


        return self
class ProvenanceMap(BaseModel):
    """
    Maps a claim / passage back to its source chain.

    Fields:
        claim_id   : identifier of the claim being traced
        chunk_ids  : ordered chain of chunk_ids (retrieval path)
        source_uri : final authoritative source URI
        hops       : number of retrieval hops traversed
    """
    claim_id: str
    chunk_ids: List[str]
    source_uri: str
    hops: int = 0


class RetrievalResult(BaseModel):
    """
    Output contract for RetrievalAgent.

    Fields:
        chunks          : all retrieved chunks across all hops
        provenance_map  : list of ProvenanceMap entries
        total_hops      : maximum hop depth reached
        metadata        : arbitrary extra info

    Raises:
        ContractValidationError: if chunks is empty
    """
    chunks: List[RetrievedChunk]
    provenance_map: List[ProvenanceMap] = Field(default_factory=list)
    total_hops: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_model(self):
        if not self.chunks:
            raise ContractValidationError("RetrievalResult.chunks must not be empty")


        return self
# ---------------------------------------------------------------------------
# CritiqueResult
# ---------------------------------------------------------------------------

class ClaimScore(BaseModel):
    """
    Confidence score for a single extracted claim.

    Fields:
        claim_id   : unique claim identifier
        claim_text : the claim being evaluated
        confidence : ConfidenceLevel enum value
        score      : numeric score 0.0 – 1.0
        rationale  : brief explanation of the score
        sources    : chunk_ids or doc_ids that support / refute this claim
    """
    claim_id: str
    claim_text: str
    confidence: ConfidenceLevel
    score: float
    rationale: str = ""
    sources: List[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_model(self):
        if not (0.0 <= self.score <= 1.0):
            raise ContractValidationError(
                f"ClaimScore.score must be in [0, 1]; got {self.score}"
            )


        return self
class Contradiction(BaseModel):
    """
    A detected contradiction between two claims.

    Fields:
        contradiction_id  : unique identifier
        claim_a_id        : first conflicting claim
        claim_b_id        : second conflicting claim
        description       : human-readable explanation
        severity          : 'high' | 'medium' | 'low'
        resolution_hint   : optional suggested resolution strategy
    """
    contradiction_id: str
    claim_a_id: str
    claim_b_id: str
    description: str
    severity: str = "medium"
    resolution_hint: str = ""

    @model_validator(mode='after')
    def validate_model(self):
        if self.severity not in {"high", "medium", "low"}:
            raise ContractValidationError(
                f"Contradiction.severity must be 'high', 'medium', or 'low'; got '{self.severity}'"
            )


        return self
class CritiqueResult(BaseModel):
    """
    Output contract for CritiqueAgent.

    Fields:
        claim_scores    : per-claim confidence evaluations
        contradictions  : detected contradictions between claims
        overall_quality : aggregate quality score 0.0 – 1.0
        metadata        : arbitrary extra info

    Raises:
        ContractValidationError: if claim_scores is empty
    """
    claim_scores: List[ClaimScore]
    contradictions: List[Contradiction] = Field(default_factory=list)
    overall_quality: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_model(self):
        if not self.claim_scores:
            raise ContractValidationError("CritiqueResult.claim_scores must not be empty")
        if not (0.0 <= self.overall_quality <= 1.0):
            raise ContractValidationError(
                f"CritiqueResult.overall_quality must be in [0, 1]; got {self.overall_quality}"
            )


        return self
# ---------------------------------------------------------------------------
# SynthesisResult
# ---------------------------------------------------------------------------

class ResolvedContradiction(BaseModel):
    """
    Records how a detected contradiction was resolved.

    Fields:
        contradiction_id : references CritiqueResult.Contradiction
        resolution       : chosen resolution strategy label
        rationale        : explanation of why this resolution was picked
        winning_claim_id : the claim that was kept (if applicable)
    """
    contradiction_id: str
    resolution: str
    rationale: str
    winning_claim_id: Optional[str] = None


class SynthesisResult(BaseModel):
    """
    Output contract for SynthesisAgent.

    Fields:
        merged_output           : final synthesized text / structured answer
        resolved_contradictions : log of how contradictions were handled
        contributing_sources    : doc_ids / chunk_ids that contributed
        confidence              : aggregate confidence of the synthesis
        metadata                : arbitrary extra info

    Raises:
        ContractValidationError: if merged_output is empty
    """
    merged_output: str
    resolved_contradictions: List[ResolvedContradiction] = Field(default_factory=list)
    contributing_sources: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_model(self):
        if not self.merged_output.strip():
            raise ContractValidationError("SynthesisResult.merged_output must not be empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ContractValidationError(
                f"SynthesisResult.confidence must be in [0, 1]; got {self.confidence}"
            )


        return self
# ---------------------------------------------------------------------------
# CompressionResult
# ---------------------------------------------------------------------------

class StructuredBlock(BaseModel):
    """
    A losslessly preserved structured block.

    Fields:
        block_id      : unique identifier within the run
        block_type    : 'table' | 'code' | 'json' | 'list' | 'formula' | other
        original      : exact original representation (MUST be preserved verbatim)
        position_hint : approximate position in the original text (char offset)
    """
    block_id: str
    block_type: str
    original: str
    position_hint: int = 0


class CompressionResult(BaseModel):
    """
    Output contract for CompressionAgent.

    Fields:
        compressed_text    : summary of conversational filler + narrative text
        structured_blocks  : losslessly preserved structured data blocks
        compression_ratio  : len(compressed) / len(original); < 1.0 means reduced
        tokens_saved       : estimated token count reduction
        metadata           : arbitrary extra info

    Raises:
        ContractValidationError: if compressed_text is empty
                                 or compression_ratio is non-positive
    """
    compressed_text: str
    structured_blocks: List[StructuredBlock] = Field(default_factory=list)
    compression_ratio: float = 1.0
    tokens_saved: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_model(self):
        if not self.compressed_text.strip():
            raise ContractValidationError("CompressionResult.compressed_text must not be empty")
        if self.compression_ratio <= 0.0:
            raise ContractValidationError(
                f"CompressionResult.compression_ratio must be > 0; got {self.compression_ratio}"
            )
        return self
