"""
tools/self_reflection_tool.py
=============================
Analyses previous tool outputs stored in SharedContext, identifies
logical contradictions, and returns a structured reflection report.

Imports
-------
    stdlib  : logging, re, time, itertools
    internal: interfaces.base_tool.BaseTool
              contracts.tool_contracts.{ToolRequest, ToolResponse,
                  ToolStatus, ReflectionResult}
              contracts.shared_context.SharedContext, AgentOutput

Inputs  : ToolRequest
    payload keys:
        scope          (str, default="all")  — "all" | tool_name filter
        step_ids       (list[str], optional) — restrict to specific steps
        sensitivity    (float, 0.0–1.0, default=0.5)
                       — contradiction detection threshold
        include_raw    (bool, default=False) — attach raw output refs

Outputs : ToolResponse
    data: ReflectionResult
        contradictions   : list of dicts describing each pair
        summary          : human-readable paragraph
        confidence_score : overall consistency score (1.0 = fully consistent)
        flagged_segments : list of problematic text snippets

Exceptions handled
------------------
    ValueError   — bad sensitivity, unknown step_id
    TypeError    — payload not a dict, step_ids not a list
    TimeoutError — re-raised to trigger BaseTool retry
    Exception    — catch-all retried by BaseTool

Dependencies
------------
    stdlib only.
    Swap _detect_contradictions() with an LLM-based checker for
    production-grade semantic analysis.
"""

from __future__ import annotations

import itertools
import logging
import re
import time
from typing import Any

from interfaces.base_tool import BaseTool
from contracts.tool_contracts import (
    ReflectionResult,
    ToolRequest,
    ToolResponse,
    ToolStatus,
)
from contracts.shared_context import AgentOutput, SharedContext

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Heuristic contradiction detector
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that signal potential negation or reversal
_NEGATION_WORDS = re.compile(
    r"\b(not|no|never|cannot|can't|won't|isn't|aren't|doesn't|didn't|failed|error|"
    r"false|incorrect|invalid|refused|rejected|absent|missing)\b",
    re.I,
)

_POSITIVE_WORDS = re.compile(
    r"\b(success|succeeded|found|valid|correct|true|passed|present|confirmed|ok|200)\b",
    re.I,
)


def _sentiment(text: str) -> int:
    """Returns +1 (positive), -1 (negative), or 0 (neutral)."""
    neg = len(_NEGATION_WORDS.findall(text))
    pos = len(_POSITIVE_WORDS.findall(text))
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


def _shared_keywords(a: str, b: str) -> set[str]:
    """Return meaningful keywords common to both summaries."""
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "it", "in", "of", "to", "for"}
    tokens_a  = {w.lower() for w in re.findall(r"\w+", a) if w.lower() not in stopwords and len(w) > 3}
    tokens_b  = {w.lower() for w in re.findall(r"\w+", b) if w.lower() not in stopwords and len(w) > 3}
    return tokens_a & tokens_b


def _detect_contradictions(
    outputs: list[AgentOutput],
    sensitivity: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Compare every pair of outputs.  A contradiction is flagged when two
    outputs share keywords but carry opposite sentiment and the keyword
    overlap ratio exceeds *sensitivity*.

    Returns (contradictions_list, flagged_segments).

    In production, replace this function body with an LLM call:
        "Given these two summaries, do they contradict each other? ..."
    """
    contradictions: list[dict[str, Any]] = []
    flagged: list[str] = []

    for a, b in itertools.combinations(outputs, 2):
        shared = _shared_keywords(a.summary, b.summary)
        if not shared:
            continue

        # Overlap ratio relative to the smaller summary's vocabulary
        all_a = {w.lower() for w in re.findall(r"\w+", a.summary) if len(w) > 3}
        all_b = {w.lower() for w in re.findall(r"\w+", b.summary) if len(w) > 3}
        denom = min(len(all_a), len(all_b)) or 1
        overlap_ratio = len(shared) / denom

        if overlap_ratio < sensitivity:
            continue

        sent_a = _sentiment(a.summary)
        sent_b = _sentiment(b.summary)

        if sent_a != 0 and sent_b != 0 and sent_a != sent_b:
            detail = {
                "step_a"        : a.step_id,
                "step_b"        : b.step_id,
                "tool_a"        : a.tool_name,
                "tool_b"        : b.tool_name,
                "shared_keywords": sorted(shared),
                "overlap_ratio" : round(overlap_ratio, 3),
                "summary_a"     : a.summary,
                "summary_b"     : b.summary,
            }
            contradictions.append(detail)
            flagged.append(a.summary[:120])
            flagged.append(b.summary[:120])

    return contradictions, flagged


# ─────────────────────────────────────────────────────────────────────────────
#  Tool
# ─────────────────────────────────────────────────────────────────────────────

class SelfReflectionTool(BaseTool):
    """
    Reads all previous tool outputs from SharedContext and produces a
    structured contradiction / consistency report.
    """

    TOOL_NAME      : str   = "self_reflection"
    VERSION        : str   = "1.0.0"
    MAX_RETRIES    : int   = 2
    TIMEOUT_SECONDS: float = 10.0

    # ── validation ────────────────────────────────────────────────────── #

    def validate(self, request: ToolRequest) -> None:
        if not isinstance(request.payload, dict):
            raise TypeError(f"payload must be dict, got {type(request.payload).__name__}")

        sensitivity = request.payload.get("sensitivity", 0.5)
        if not isinstance(sensitivity, float | int) or not (0.0 <= float(sensitivity) <= 1.0):
            raise ValueError("payload.sensitivity must be a float in [0.0, 1.0]")

        step_ids = request.payload.get("step_ids")
        if step_ids is not None and not isinstance(step_ids, list):
            raise TypeError("payload.step_ids must be a list of strings or None")

    # ── execution ─────────────────────────────────────────────────────── #

    def execute(self, request: ToolRequest, context: SharedContext) -> ToolResponse:
        t0          = time.monotonic()
        scope       = request.payload.get("scope", "all")
        step_ids    = request.payload.get("step_ids")          # list[str] | None
        sensitivity = float(request.payload.get("sensitivity", 0.5))

        # ── collect outputs from context ─────────────────────────────── #
        if scope == "all":
            outputs = context.all_outputs()
        else:
            outputs = context.get_outputs_by_tool(scope)

        if step_ids:
            try:
                outputs = [context.get_output(sid) for sid in step_ids]
            except KeyError as exc:
                raise ValueError(str(exc)) from exc

        if not outputs:
            return ToolResponse.failure(
                request_id=request.request_id,
                tool_name =self.TOOL_NAME,
                status    =ToolStatus.EMPTY,
                error     ="No previous outputs found in context to reflect on",
            )

        logger.debug(
            "[%s] reflecting on %d output(s) | sensitivity=%.2f",
            self.TOOL_NAME, len(outputs), sensitivity,
        )

        # ── detect contradictions ─────────────────────────────────────── #
        contradictions, flagged = _detect_contradictions(outputs, sensitivity)

        n_outputs = len(outputs)
        n_contradictions = len(contradictions)

        # Confidence: 1.0 when no contradictions; degrades with each pair
        max_pairs = max((n_outputs * (n_outputs - 1)) // 2, 1)
        confidence = round(1.0 - (n_contradictions / max_pairs), 4)
        confidence = max(0.0, min(1.0, confidence))

        summary_parts = [
            f"Reflected on {n_outputs} output(s) across "
            f"{len({o.tool_name for o in outputs})} tool(s)."
        ]
        if n_contradictions == 0:
            summary_parts.append("No contradictions detected. Outputs appear consistent.")
        else:
            summary_parts.append(
                f"Detected {n_contradictions} potential contradiction(s). "
                f"Consistency score: {confidence:.2%}. "
                "Review flagged segments before using these outputs."
            )

        result = ReflectionResult(
            contradictions  =contradictions,
            summary         =" ".join(summary_parts),
            confidence_score=confidence,
            flagged_segments=list(dict.fromkeys(flagged)),  # dedupe, preserve order
        )

        return ToolResponse.success(
            request_id =request.request_id,
            tool_name  =self.TOOL_NAME,
            data       =result,
            duration_ms=(time.monotonic() - t0) * 1000,
        )
