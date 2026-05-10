"""
agents/synthesis_agent.py
==========================
SynthesisAgent — merges outputs from prior pipeline agents, resolves
contradictions identified by the CritiqueAgent, and produces a single
coherent answer with attributed contributing sources.

SOLID Alignment:
  - (S) Responsible only for merging and contradiction resolution
  - (O) Override _merge_strategy() to swap in an LLM-backed synthesis
  - (L) Safe wherever BaseAgent[SharedContext, SynthesisResult] is expected
  - (D) Depends on interfaces/contracts; reads peer outputs via SharedContext

Imports (external):
  stdlib  : logging, re, uuid
  local   : interfaces.base_agent, contracts.shared_context,
            contracts.agent_contracts

Input:
  SharedContext
    .query         — user query for framing the output (required)
    .documents     — optional additional document corpus
    .agent_outputs — expects at least one of:
                       'RetrievalAgent'    → RetrievalResult
                       'CritiqueAgent'     → CritiqueResult
    .metadata      — optional hints:
                       'max_output_tokens' (int,   default 800)
                       'resolution_strategy' (str, default 'highest_confidence')

Output:
  SynthesisResult
    .merged_output           — final coherent answer string
    .resolved_contradictions — log of contradiction resolutions
    .contributing_sources    — list of source identifiers used
    .confidence              — aggregate confidence 0.0 – 1.0
    .metadata                — diagnostics

Exceptions:
  AgentValidationError   : query empty or no usable content found
  AgentExecutionError    : merge or resolution logic fails
  ContractValidationError: output invariant violation (empty merged_output)

Dependencies:
  interfaces.base_agent.BaseAgent
  contracts.shared_context.SharedContext
  contracts.agent_contracts.SynthesisResult, ResolvedContradiction,
                             RetrievalResult, CritiqueResult, ClaimScore
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Dict, List, Optional, Tuple

from interfaces.base_agent import BaseAgent, AgentExecutionError, AgentValidationError
from contracts.shared_context import SharedContext
from contracts.agent_contracts import (
    ClaimScore,
    ConfidenceLevel,
    ContractValidationError,
    CritiqueResult,
    ResolvedContradiction,
    RetrievalResult,
    SynthesisResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolution strategies
# ---------------------------------------------------------------------------

_RESOLUTION_STRATEGIES = frozenset({
    "highest_confidence",   # keep the claim with the higher score
    "most_recent_source",   # prefer the source with the larger doc_id (proxy)
    "merge_both",           # include both claims with a note
})


def _resolve_by_confidence(
    claim_a: ClaimScore,
    claim_b: ClaimScore,
    contra_id: str,
) -> ResolvedContradiction:
    winner = claim_a if claim_a.score >= claim_b.score else claim_b
    return ResolvedContradiction(
        contradiction_id=contra_id,
        resolution="highest_confidence",
        rationale=(
            f"Claim '{winner.claim_id}' retained (score={winner.score:.3f}) "
            f"over lower-confidence alternative (score={min(claim_a.score, claim_b.score):.3f})"
        ),
        winning_claim_id=winner.claim_id,
    )


def _resolve_merge_both(
    claim_a: ClaimScore,
    claim_b: ClaimScore,
    contra_id: str,
) -> ResolvedContradiction:
    return ResolvedContradiction(
        contradiction_id=contra_id,
        resolution="merge_both",
        rationale="Both perspectives included to reflect uncertainty",
        winning_claim_id=None,
    )


# ---------------------------------------------------------------------------
# SynthesisAgent
# ---------------------------------------------------------------------------

class SynthesisAgent(BaseAgent[SharedContext, SynthesisResult]):
    """
    Merges multi-agent outputs into a coherent final answer.

    Pipeline:
      1. Collect passages from RetrievalResult and/or raw documents.
      2. Resolve contradictions flagged by CritiqueResult.
      3. Assemble the merged_output with contributing source references.
      4. Compute aggregate confidence.

    Override _merge_strategy() to integrate an LLM summarizer.
    """

    TIMEOUT_SECONDS = 45.0

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def validate_input(self, context: SharedContext) -> None:
        super().validate_input(context)
        if not context.query.strip():
            raise AgentValidationError(self.agent_name, "SharedContext.query must not be empty")

    def validate_output(self, result: SynthesisResult) -> None:
        super().validate_output(result)
        if not result.merged_output.strip():
            raise AgentValidationError(self.agent_name, "SynthesisResult.merged_output must not be empty")

    # ------------------------------------------------------------------ #
    #  Core run                                                            #
    # ------------------------------------------------------------------ #

    def run(self, context: SharedContext) -> SynthesisResult:
        """
        Merge and synthesize pipeline outputs.

        Args:
            context: SharedContext with prior agent outputs

        Returns:
            SynthesisResult

        Raises:
            AgentExecutionError: if synthesis fails
        """
        logger.info("Starting synthesis | run_id=%s", context.run_id)

        resolution_strategy: str = context.metadata.get(
            "resolution_strategy", "highest_confidence"
        )
        if resolution_strategy not in _RESOLUTION_STRATEGIES:
            logger.warning(
                "Unknown resolution_strategy '%s'; defaulting to 'highest_confidence'",
                resolution_strategy,
            )
            resolution_strategy = "highest_confidence"

        try:
            retrieval: Optional[RetrievalResult] = context.get_agent_output("RetrievalAgent")
            critique:  Optional[CritiqueResult]  = context.get_agent_output("CritiqueAgent")

            # --- collect content ---
            passages, sources = self._collect_passages(context, retrieval, critique)

            _bare_query_only = (
                len(passages) == 1
                and passages[0] == context.query
                and sources == ["query"]
                and not context.documents
            )
            if not passages or _bare_query_only:
                raise AgentExecutionError(
                    self.agent_name,
                    "No content available for synthesis (no documents, retrieval, or critique output)",
                )

            # --- resolve contradictions ---
            resolved = self._resolve_contradictions(critique, resolution_strategy)

            # --- build merged output ---
            merged = self._merge_strategy(context.query, passages, resolved, critique)

            # --- aggregate confidence ---
            confidence = self._aggregate_confidence(critique, retrieval)

        except AgentExecutionError:
            raise
        except Exception as exc:
            raise AgentExecutionError(self.agent_name, f"Synthesis failed: {exc}") from exc

        result = SynthesisResult(
            merged_output=merged,
            resolved_contradictions=resolved,
            contributing_sources=list(dict.fromkeys(sources)),  # preserve order, dedupe
            confidence=round(confidence, 4),
            metadata={
                "run_id":              context.run_id,
                "passages_used":       len(passages),
                "contradictions_resolved": len(resolved),
                "resolution_strategy": resolution_strategy,
            },
        )
        logger.info(
            "Synthesis complete | passages=%d resolved=%d confidence=%.3f",
            len(passages), len(resolved), confidence,
        )
        context.store_agent_output(self.agent_name, result)
        return result

    # ------------------------------------------------------------------ #
    #  Content collection                                                  #
    # ------------------------------------------------------------------ #

    def _collect_passages(
        self,
        context: SharedContext,
        retrieval: Optional[RetrievalResult],
        critique:  Optional[CritiqueResult],
    ) -> Tuple[List[str], List[str]]:
        """
        Return (passages, sources).

        Priority:
          1. High/medium-confidence claims from CritiqueResult
          2. Retrieved chunks
          3. Raw document content
          4. Query itself as last resort
        """
        passages: List[str] = []
        sources:  List[str] = []

        if critique and critique.claim_scores:
            for cs in critique.claim_scores:
                if cs.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM):
                    passages.append(cs.claim_text)
                    sources.extend(cs.sources)

        if not passages and retrieval and retrieval.chunks:
            for chunk in sorted(retrieval.chunks, key=lambda c: c.score, reverse=True):
                passages.append(chunk.content)
                sources.append(chunk.source)

        if not passages and context.documents:
            for doc in context.documents:
                passages.append(doc.content)
                sources.append(doc.source)

        if not passages:
            passages = [context.query]
            sources  = ["query"]

        return passages, sources

    # ------------------------------------------------------------------ #
    #  Contradiction resolution                                           #
    # ------------------------------------------------------------------ #

    def _resolve_contradictions(
        self,
        critique: Optional[CritiqueResult],
        strategy: str,
    ) -> List[ResolvedContradiction]:
        if not critique or not critique.contradictions:
            return []

        resolved: List[ResolvedContradiction] = []
        claim_index: Dict[str, ClaimScore] = {
            cs.claim_id: cs for cs in critique.claim_scores
        }

        for contra in critique.contradictions:
            cs_a = claim_index.get(contra.claim_a_id)
            cs_b = claim_index.get(contra.claim_b_id)
            if not cs_a or not cs_b:
                continue

            if strategy == "merge_both":
                resolved.append(_resolve_merge_both(cs_a, cs_b, contra.contradiction_id))
            else:  # highest_confidence (default) and most_recent_source fall back here
                resolved.append(_resolve_confidence(cs_a, cs_b, contra.contradiction_id))

        return resolved

    # ------------------------------------------------------------------ #
    #  Override-friendly merge strategy                                   #
    # ------------------------------------------------------------------ #

    def _merge_strategy(
        self,
        query: str,
        passages: List[str],
        resolved: List[ResolvedContradiction],
        critique: Optional[CritiqueResult],
    ) -> str:
        """
        Build the final merged output string.

        Override to integrate an LLM summarizer.

        Args:
            query    : original user query
            passages : ordered list of content passages to merge
            resolved : contradiction resolutions already applied
            critique : full CritiqueResult (may be None)

        Returns:
            Non-empty merged output string
        """
        lines: List[str] = []
        lines.append(f"Query: {query}\n")
        lines.append("Synthesized Answer:")
        lines.append("-" * 40)

        # De-duplicate and truncate long passages
        seen: set = set()
        for passage in passages:
            key = passage[:120]
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"• {passage.strip()}")

        if resolved:
            lines.append("\nContradiction Resolutions:")
            for res in resolved:
                lines.append(
                    f"  [{res.contradiction_id}] {res.resolution} — {res.rationale}"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Aggregate confidence                                               #
    # ------------------------------------------------------------------ #

    def _aggregate_confidence(
        self,
        critique:  Optional[CritiqueResult],
        retrieval: Optional[RetrievalResult],
    ) -> float:
        if critique and critique.overall_quality > 0:
            return critique.overall_quality
        if retrieval and retrieval.chunks:
            scores = [c.score for c in retrieval.chunks]
            return sum(scores) / len(scores)
        return 0.5  # neutral default


# Resolution helper alias (avoids shadowing the method)
def _resolve_confidence(a: ClaimScore, b: ClaimScore, cid: str) -> ResolvedContradiction:
    return _resolve_by_confidence(a, b, cid)


# ---------------------------------------------------------------------------
# Debug entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        stream=sys.stdout,
    )

    print("=" * 60)
    print("SynthesisAgent — debug run")
    print("=" * 60)

    # Build a context with pre-populated agent_outputs to simulate pipeline
    from contracts.agent_contracts import (
        RetrievedChunk, ProvenanceMap,
        ClaimScore, ConfidenceLevel, Contradiction,
    )

    ctx = SharedContext(
        query="What are efficient transformer attention mechanisms?",
        metadata={"resolution_strategy": "highest_confidence"},
    )

    # Simulate RetrievalAgent output
    r_result = RetrievalResult(
        chunks=[
            RetrievedChunk(
                chunk_id="c1", doc_id="d1", source="paper_a",
                content="FlashAttention is a fast, memory-efficient exact attention algorithm.",
                score=0.85, hop=0,
            ),
            RetrievedChunk(
                chunk_id="c2", doc_id="d2", source="paper_b",
                content="Linear attention is not always as accurate as standard attention.",
                score=0.60, hop=1,
            ),
        ],
        total_hops=1,
    )
    ctx.store_agent_output("RetrievalAgent", r_result)

    # Simulate CritiqueAgent output
    cs1 = ClaimScore(claim_id="cl1", claim_text=r_result.chunks[0].content,
                     confidence=ConfidenceLevel.HIGH, score=0.85)
    cs2 = ClaimScore(claim_id="cl2", claim_text=r_result.chunks[1].content,
                     confidence=ConfidenceLevel.MEDIUM, score=0.60)
    c_result = CritiqueResult(
        claim_scores=[cs1, cs2],
        contradictions=[
            Contradiction(
                contradiction_id="contra_001",
                claim_a_id="cl1", claim_b_id="cl2",
                description="Efficiency claim vs accuracy caveat",
                severity="medium",
            )
        ],
        overall_quality=0.72,
    )
    ctx.store_agent_output("CritiqueAgent", c_result)

    agent  = SynthesisAgent()
    result = agent(ctx)

    print("\n--- Merged Output ---")
    print(result.merged_output)
    print("\n--- Resolved Contradictions ---")
    for res in result.resolved_contradictions:
        print(f"  [{res.contradiction_id}] {res.resolution}: {res.rationale}")
    print(f"\n--- Contributing Sources: {result.contributing_sources} ---")
    print(f"--- Confidence: {result.confidence} ---")
    print("\n--- Metadata ---")
    print(json.dumps(result.metadata, indent=2))
    print("\nSchema validation: PASSED ✓")
