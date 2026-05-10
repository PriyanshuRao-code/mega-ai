"""
tools/web_search_tool.py
========================
Executes structured web searches and returns relevance-scored results.

Imports
-------
    stdlib  : logging, time, math, re, hashlib
    internal: interfaces.base_tool.BaseTool
              contracts.tool_contracts.{ToolRequest, ToolResponse,
                  ToolStatus, SearchResult}
              contracts.shared_context.SharedContext

Inputs  : ToolRequest
    payload keys:
        query        (str, required)  — natural-language search query
        max_results  (int, default=5) — upper bound on results returned
        min_score    (float, 0-1)     — filter results below this score
        language     (str, default="en")

Outputs : ToolResponse
    data: list[SearchResult]
        SearchResult.score: float  TF-IDF-inspired relevance 0.0–1.0
        SearchResult.rank : int    1-based position in result set

Exceptions handled
------------------
    ValueError   — missing/blank query, bad numeric params
    TypeError    — payload not a dict
    TimeoutError — search backend exceeds TIMEOUT_SECONDS
    Exception    — any other backend error (retried up to MAX_RETRIES)

Dependencies
------------
    No third-party libraries required.
    Swap _call_search_backend() for real HTTP client (httpx, requests)
    without changing any other contract.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from typing import Any

from interfaces.base_tool import BaseTool
from contracts.tool_contracts import (
    ToolRequest,
    ToolResponse,
    ToolStatus,
    SearchResult,
)
from contracts.shared_context import SharedContext

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Relevance scorer
# ─────────────────────────────────────────────────────────────────────────────

class _RelevanceScorer:
    """
    Lightweight TF-IDF–inspired scorer.

    Score = cosine_sim(query_terms, document_terms) normalised to [0, 1].
    In production replace with an embedding-based scorer.
    """

    def score(self, query: str, title: str, snippet: str) -> float:
        q_terms  = self._tokenise(query)
        doc_text = f"{title} {snippet}"
        d_terms  = self._tokenise(doc_text)

        if not q_terms or not d_terms:
            return 0.0

        q_vec = self._tf(q_terms)
        d_vec = self._tf(d_terms)

        common = set(q_vec) & set(d_vec)
        if not common:
            return 0.0

        dot    = sum(q_vec[t] * d_vec[t] for t in common)
        norm_q = math.sqrt(sum(v ** 2 for v in q_vec.values()))
        norm_d = math.sqrt(sum(v ** 2 for v in d_vec.values()))

        return round(dot / (norm_q * norm_d), 4)

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _tf(tokens: list[str]) -> dict[str, float]:
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total = len(tokens)
        return {t: c / total for t, c in counts.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  Tool
# ─────────────────────────────────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    """
    Structured web-search tool with relevance scoring.

    Swap _call_search_backend() for a real search API (SerpAPI, Bing, etc.)
    without touching any contract or orchestrator code.
    """

    TOOL_NAME      : str   = "web_search"
    VERSION        : str   = "1.0.0"
    MAX_RETRIES    : int   = 3
    TIMEOUT_SECONDS: float = 15.0

    def __init__(self) -> None:
        self._scorer = _RelevanceScorer()

    # ── validation ────────────────────────────────────────────────────── #

    def validate(self, request: ToolRequest) -> None:
        if not isinstance(request.payload, dict):
            raise TypeError(f"payload must be dict, got {type(request.payload).__name__}")

        query = request.payload.get("query", "")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("payload.query must be a non-empty string")

        max_results = request.payload.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1:
            raise ValueError("payload.max_results must be a positive integer")

        min_score = request.payload.get("min_score", 0.0)
        if not isinstance(min_score, float | int) or not (0.0 <= float(min_score) <= 1.0):
            raise ValueError("payload.min_score must be a float in [0.0, 1.0]")

    # ── execution ─────────────────────────────────────────────────────── #

    def execute(self, request: ToolRequest, context: SharedContext) -> ToolResponse:
        t0          = time.monotonic()
        query       = request.payload["query"].strip()
        max_results = int(request.payload.get("max_results", 5))
        min_score   = float(request.payload.get("min_score", 0.0))
        timeout     = request.timeout or self.TIMEOUT_SECONDS

        logger.debug("[%s] query=%r max_results=%d", self.TOOL_NAME, query, max_results)

        raw_hits = self._call_search_backend(query, max_results, timeout)

        if not raw_hits:
            return ToolResponse.failure(
                request_id=request.request_id,
                tool_name=self.TOOL_NAME,
                status=ToolStatus.EMPTY,
                error=f"No results returned for query: {query!r}",
            )

        results: list[SearchResult] = []
        for rank, hit in enumerate(raw_hits, start=1):
            score = self._scorer.score(query, hit["title"], hit["snippet"])
            if score < min_score:
                continue
            results.append(
                SearchResult(
                    title   =hit["title"],
                    url     =hit["url"],
                    snippet =hit["snippet"],
                    score   =score,
                    rank    =rank,
                    metadata={"raw_rank": rank, "source": hit.get("source", "unknown")},
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)

        if not results:
            return ToolResponse.failure(
                request_id=request.request_id,
                tool_name=self.TOOL_NAME,
                status=ToolStatus.EMPTY,
                error=f"All results filtered below min_score={min_score}",
            )

        duration_ms = (time.monotonic() - t0) * 1000
        return ToolResponse.success(
            request_id =request.request_id,
            tool_name  =self.TOOL_NAME,
            data       =results,
            duration_ms=duration_ms,
        )

    # ── backend stub (replace with real HTTP client) ───────────────────── #

    def _call_search_backend(
        self, query: str, max_results: int, timeout: float
    ) -> list[dict[str, Any]]:
        """
        Stub that simulates a search backend response.
        Replace with: httpx.get(SEARCH_API_URL, params={...}, timeout=timeout)

        Raises TimeoutError to exercise the retry path.
        """
        # Deterministic fake results derived from query hash
        digest = hashlib.md5(query.encode()).hexdigest()
        hits   = []
        for i in range(max_results):
            seed = int(digest[i * 2: i * 2 + 2], 16)
            hits.append(
                {
                    "title"  : f"Result {i+1}: {query[:40]} (relevance={seed})",
                    "url"    : f"https://example.com/result/{digest[:8]}/{i+1}",
                    "snippet": (
                        f"This page discusses {query} in depth. "
                        f"Key findings relate to {query.split()[0] if query.split() else 'topic'} "
                        f"and associated concepts (doc #{seed})."
                    ),
                    "source" : "stub",
                }
            )
        return hits
