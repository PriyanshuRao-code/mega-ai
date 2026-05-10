"""
Module: contracts/eval_contracts.py
=====================================
Purpose:
    Defines all Pydantic contracts used by the evaluation framework:
    test cases, per-metric scores, aggregate results, and regression
    tracking across pipeline versions.

    These contracts are consumed by the eval harness, CI pipelines,
    and the debug healthcheck system.

Input Datatypes:
    None (pure schema definitions)

Output Datatypes:
    - EvalCase
    - EvalScore
    - EvalResult
    - RegressionResult

Dependencies:
    - pydantic >= 2.0
    - Python 3.11+

SOLID Principle:
    Single Responsibility — this file ONLY defines evaluation contracts.
    Liskov Substitution — EvalResult can hold any list of EvalScore subtypes.
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


class EvalStatus(str, Enum):
    """Outcome of an eval case or run."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class RegressionSeverity(str, Enum):
    """How severe a detected regression is."""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Type of evaluation metric being scored."""

    ACCURACY = "accuracy"
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    FLUENCY = "fluency"
    COHERENCE = "coherence"
    CITATION_QUALITY = "citation_quality"
    LATENCY = "latency"
    TOKEN_EFFICIENCY = "token_efficiency"
    POLICY_COMPLIANCE = "policy_compliance"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# EvalCase — a single test input/expected-output pair
# ---------------------------------------------------------------------------


class EvalCase(BaseModel):
    """
    A single evaluation test case.

    Fields
    ------
    case_id : str
        Unique identifier for this test case.
    name : str
        Human-readable name of the test case.
    description : str
        What this case is testing.
    query : str
        Input query or task.
    expected_output : str | None
        Reference/golden output (may be None for human-evaluation cases).
    expected_citations : list[str]
        Expected citation URLs or IDs (empty list means citations are not checked).
    tags : list[str]
        Categorisation tags (e.g. "retrieval", "edge-case", "regression").
    agent_under_test : str | None
        Name of the specific agent being tested, or None for full-pipeline evals.
    timeout_ms : float
        Maximum allowed execution time for this case.
    metadata : dict[str, Any]
        Arbitrary extra metadata for this case.
    created_at : datetime
        UTC creation timestamp.
    """

    case_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique test case ID.",
    )
    name: str = Field(..., min_length=1, description="Human-readable test case name.")
    description: str = Field(
        default="",
        description="What this case is testing.",
    )
    query: str = Field(..., min_length=1, description="Input query or task.")
    expected_output: str | None = Field(
        default=None,
        description="Reference/golden output string.",
    )
    expected_citations: list[str] = Field(
        default_factory=list,
        description="Expected citation URLs or IDs.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Categorisation tags.",
    )
    agent_under_test: str | None = Field(
        default=None,
        description="Specific agent being tested (None = full pipeline).",
    )
    timeout_ms: float = Field(
        default=30_000.0,
        gt=0.0,
        description="Execution timeout for this case in ms.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra metadata.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC creation timestamp.",
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("EvalCase.query must not be blank.")
        return v.strip()


# ---------------------------------------------------------------------------
# EvalScore — a single metric measurement for one EvalCase
# ---------------------------------------------------------------------------


class EvalScore(BaseModel):
    """
    A single metric score produced by evaluating one EvalCase.

    Fields
    ------
    score_id : str
        Unique ID for this score.
    case_id : str
        Matching EvalCase.case_id.
    metric_type : MetricType
        The metric being measured.
    metric_name : str
        Human-readable metric label (defaults to metric_type value).
    score : float
        Numeric score (0.0–1.0 for normalised metrics; unbounded for latency).
    passing_threshold : float
        Minimum score required to pass this metric.
    passed : bool
        True iff score >= passing_threshold.
    explanation : str
        Brief explanation of why this score was assigned.
    scorer_name : str
        Name of the scorer (e.g. "LLM-judge", "exact-match", "rouge-L").
    scored_at : datetime
        UTC timestamp when scoring occurred.
    metadata : dict[str, Any]
        Extra scorer-specific metadata.
    """

    score_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique score ID.",
    )
    case_id: str = Field(..., description="Matching EvalCase.case_id.")
    metric_type: MetricType = Field(..., description="Metric being measured.")
    metric_name: str = Field(
        default="",
        description="Human-readable label (defaults to metric_type value).",
    )
    score: float = Field(..., description="Numeric score value.")
    passing_threshold: float = Field(
        default=0.7,
        description="Minimum score to pass this metric.",
    )
    passed: bool = Field(
        default=False,
        description="True iff score >= passing_threshold.",
    )
    explanation: str = Field(
        default="",
        description="Brief explanation of the score.",
    )
    scorer_name: str = Field(
        default="unknown",
        description="Name of the scorer.",
    )
    scored_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC scoring timestamp.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra scorer-specific metadata.",
    )

    @model_validator(mode="after")
    def derive_passed_and_name(self) -> "EvalScore":
        self.passed = self.score >= self.passing_threshold
        if not self.metric_name:
            self.metric_name = self.metric_type.value
        return self


# ---------------------------------------------------------------------------
# EvalResult — aggregate result for one EvalCase
# ---------------------------------------------------------------------------


class EvalResult(BaseModel):
    """
    Aggregate evaluation result for a single EvalCase run.

    Fields
    ------
    result_id : str
        Unique ID for this result.
    case_id : str
        Matching EvalCase.case_id.
    job_id : str
        Pipeline run that was evaluated.
    status : EvalStatus
        Overall outcome of this eval case.
    scores : list[EvalScore]
        All metric scores for this case.
    overall_score : float
        Mean of all metric scores (computed automatically).
    overall_passed : bool
        True iff ALL scores passed their thresholds.
    actual_output : str | None
        The actual output produced by the pipeline.
    actual_citations : list[str]
        Citations actually produced.
    latency_ms : float
        Actual execution latency for this case.
    error_message : str | None
        Set if status is ERROR.
    evaluated_at : datetime
        UTC evaluation timestamp.
    pipeline_version : str
        Version tag of the pipeline under test.
    metadata : dict[str, Any]
        Arbitrary extra metadata.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique result ID.",
    )
    case_id: str = Field(..., description="Matching EvalCase.case_id.")
    job_id: str = Field(..., description="Pipeline run that was evaluated.")
    status: EvalStatus = Field(default=EvalStatus.PASSED)
    scores: list[EvalScore] = Field(
        default_factory=list,
        description="All metric scores for this case.",
    )
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Mean of all metric scores (derived).",
    )
    overall_passed: bool = Field(
        default=False,
        description="True iff ALL scores passed their thresholds.",
    )
    actual_output: str | None = Field(
        default=None,
        description="Actual output produced by the pipeline.",
    )
    actual_citations: list[str] = Field(
        default_factory=list,
        description="Citations actually produced.",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Actual execution latency in ms.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error description if status is ERROR.",
    )
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC evaluation timestamp.",
    )
    pipeline_version: str = Field(
        default="unknown",
        description="Version tag of the pipeline under test.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra metadata.",
    )

    @model_validator(mode="after")
    def derive_overall(self) -> "EvalResult":
        if self.scores:
            self.overall_score = round(
                sum(s.score for s in self.scores) / len(self.scores), 4
            )
            self.overall_passed = all(s.passed for s in self.scores)
        return self

    def score_by_metric(self, metric: MetricType) -> EvalScore | None:
        """Return the EvalScore for a specific metric type, or None."""
        for s in self.scores:
            if s.metric_type == metric:
                return s
        return None


# ---------------------------------------------------------------------------
# RegressionResult — compares two pipeline versions
# ---------------------------------------------------------------------------


class RegressionResult(BaseModel):
    """
    Tracks metric changes between a baseline pipeline version and a candidate.

    Fields
    ------
    regression_id : str
        Unique ID for this comparison.
    baseline_version : str
        Version tag of the reference (previous) pipeline.
    candidate_version : str
        Version tag of the pipeline being validated.
    case_id : str
        EvalCase this comparison is based on.
    metric_type : MetricType
        The metric being compared.
    baseline_score : float
        Score achieved by the baseline pipeline.
    candidate_score : float
        Score achieved by the candidate pipeline.
    delta : float
        Signed difference: candidate_score − baseline_score.
    relative_change_pct : float
        Percentage change relative to baseline (0 if baseline is 0).
    regression_threshold : float
        Minimum delta that constitutes a regression (negative value).
        E.g. -0.05 means a drop >5% is a regression.
    is_regression : bool
        True iff delta < regression_threshold.
    severity : RegressionSeverity
        Severity band of the detected regression (or NONE).
    notes : str
        Human-readable notes about this comparison.
    compared_at : datetime
        UTC timestamp when this comparison was run.
    """

    regression_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique regression comparison ID.",
    )
    baseline_version: str = Field(..., description="Reference pipeline version tag.")
    candidate_version: str = Field(
        ..., description="Candidate pipeline version tag."
    )
    case_id: str = Field(..., description="EvalCase this comparison is based on.")
    metric_type: MetricType = Field(..., description="Metric being compared.")
    baseline_score: float = Field(..., description="Baseline pipeline score.")
    candidate_score: float = Field(..., description="Candidate pipeline score.")
    delta: float = Field(
        default=0.0,
        description="candidate_score − baseline_score (derived).",
    )
    relative_change_pct: float = Field(
        default=0.0,
        description="Percentage change relative to baseline (derived).",
    )
    regression_threshold: float = Field(
        default=-0.05,
        le=0.0,
        description="Delta below which a regression is declared (must be ≤ 0).",
    )
    is_regression: bool = Field(
        default=False,
        description="True iff delta < regression_threshold.",
    )
    severity: RegressionSeverity = Field(
        default=RegressionSeverity.NONE,
        description="Severity of detected regression.",
    )
    notes: str = Field(default="", description="Human-readable comparison notes.")
    compared_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC comparison timestamp.",
    )

    @model_validator(mode="after")
    def derive_regression_fields(self) -> "RegressionResult":
        self.delta = round(self.candidate_score - self.baseline_score, 6)
        if self.baseline_score != 0.0:
            self.relative_change_pct = round(
                (self.delta / abs(self.baseline_score)) * 100.0, 2
            )
        self.is_regression = self.delta < self.regression_threshold
        if self.is_regression:
            abs_drop = abs(self.delta)
            if abs_drop >= 0.20:
                self.severity = RegressionSeverity.CRITICAL
            elif abs_drop >= 0.10:
                self.severity = RegressionSeverity.SEVERE
            elif abs_drop >= 0.05:
                self.severity = RegressionSeverity.MODERATE
            else:
                self.severity = RegressionSeverity.MINOR
        else:
            self.severity = RegressionSeverity.NONE
        return self


# ---------------------------------------------------------------------------
# Standalone debug entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("eval_contracts.py — standalone debug mode")
    print("=" * 60)

    # EvalCase
    case = EvalCase(
        name="Capital of France",
        description="Tests that the pipeline correctly answers a simple factual question.",
        query="What is the capital of France?",
        expected_output="Paris",
        expected_citations=["https://en.wikipedia.org/wiki/France"],
        tags=["factual", "geography", "baseline"],
        agent_under_test="retrieval_agent",
    )
    print("\n[EvalCase]", case.name, case.case_id[:8])

    # EvalScore
    score1 = EvalScore(
        case_id=case.case_id,
        metric_type=MetricType.ACCURACY,
        score=0.95,
        explanation="Answer matches expected output exactly.",
        scorer_name="exact-match",
    )
    score2 = EvalScore(
        case_id=case.case_id,
        metric_type=MetricType.FAITHFULNESS,
        score=0.62,
        passing_threshold=0.70,
        explanation="Partially faithful; missing historical context.",
        scorer_name="LLM-judge",
    )
    print(f"\n[EvalScore accuracy] passed={score1.passed} score={score1.score}")
    print(f"[EvalScore faithfulness] passed={score2.passed} score={score2.score}")

    # EvalResult
    result = EvalResult(
        case_id=case.case_id,
        job_id="job-abc",
        scores=[score1, score2],
        actual_output="Paris",
        actual_citations=["https://en.wikipedia.org/wiki/France"],
        latency_ms=412.0,
        pipeline_version="v1.3.0",
    )
    print(f"\n[EvalResult] overall_score={result.overall_score} passed={result.overall_passed}")
    print(f"  status={result.status}")

    # RegressionResult — no regression
    rr_ok = RegressionResult(
        baseline_version="v1.2.0",
        candidate_version="v1.3.0",
        case_id=case.case_id,
        metric_type=MetricType.ACCURACY,
        baseline_score=0.90,
        candidate_score=0.95,
    )
    print(f"\n[RegressionResult +improvement] delta={rr_ok.delta} is_regression={rr_ok.is_regression}")

    # RegressionResult — critical regression
    rr_bad = RegressionResult(
        baseline_version="v1.2.0",
        candidate_version="v1.3.0",
        case_id=case.case_id,
        metric_type=MetricType.FAITHFULNESS,
        baseline_score=0.90,
        candidate_score=0.65,
    )
    print(f"[RegressionResult regression] delta={rr_bad.delta} severity={rr_bad.severity}")

    print("\n✅ eval_contracts.py debug complete.")