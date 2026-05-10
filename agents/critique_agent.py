"""
agents/critique_agent.py
=========================
CritiqueAgent — performs claim-level confidence scoring and contradiction
detection over retrieved and synthesized content stored in SharedContext.

SOLID Alignment:
  - (S) Responsible only for evaluation / critique logic
  - (O) Override _score_claim() / _detect_contradictions() for LLM backends
  - (L) Safe wherever BaseAgent[SharedContext, CritiqueResult] is expected
  - (D) Depends on interfaces/contracts, not on concrete agents

Imports (external):
  stdlib  : logging, re, uuid, itertools
  local   : interfaces.base_agent, contracts.shared_context,
            contracts.agent_contracts

Input:
  SharedContext
    .query           — original user query (required)
    .documents       — corpus used for grounding claims (optional)
    .agent_outputs   — expects 'RetrievalAgent' key containing RetrievalResult
                       (falls back to raw document content if absent)
    .metadata        — optional hints:
                         'min_confidence_threshold' (float, default 0.3)
                         'max_claims'               (int,   default 20)

Output:
  CritiqueResult
    .claim_scores    — per-claim confidence evaluations
    .contradictions  — detected contradictions with severity labels
    .overall_quality — aggregate quality score 0.0 – 1.0
    .metadata        — diagnostics

Exceptions:
  AgentValidationError  : query empty or context malformed
  AgentExecutionError   : claim extraction or scoring fails
  ContractValidationError: output invariant violation (no claim_scores)

Dependencies:
  interfaces.base_agent.BaseAgent
  contracts.shared_context.SharedContext
  contracts.agent_contracts.CritiqueResult, ClaimScore, Contradiction,
                             ConfidenceLevel, RetrievalResult
"""

from __future__ import annotations

import itertools
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
    Contradiction,
    CritiqueResult,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Claim extraction helpers
# ---------------------------------------------------------------------------

# Split on sentence boundaries
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Signals that lower confidence: hedging language
_HEDGE_RE = re.compile(
    r"\b(might|may|could|possibly|perhaps|allegedly|reportedly|unclear|uncertain|suggest)\b",
    re.I,
)
# Signals that raise confidence: assertive language
_ASSERT_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|will|shows|demonstrates|confirms|proves)\b",
    re.I,
)
# Contradiction signals between two sentences
_NEGATION_RE = re.compile(r"\b(not|no|never|neither|nor|cannot|won't|doesn't|don't|isn't|aren't)\b", re.I)


def _extract_sentences(text: str) -> List[str]:
    sentences = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _confidence_score(sentence: str) -> Tuple[float, ConfidenceLevel, str]:
    """
    Heuristic confidence scoring for a sentence.

    Returns:
        (score 0–1, ConfidenceLevel, rationale string)
    """
    score = 0.5  # neutral baseline

    hedges  = len(_HEDGE_RE.findall(sentence))
    asserts = len(_ASSERT_RE.findall(sentence))

    score -= hedges  * 0.08
    score += asserts * 0.05
    score  = max(0.0, min(1.0, score))

    if score >= 0.70:
        level, rationale = ConfidenceLevel.HIGH,   "Strong assertive language"
    elif score >= 0.45:
        level, rationale = ConfidenceLevel.MEDIUM, "Mixed or neutral language"
    elif score >= 0.20:
        level, rationale = ConfidenceLevel.LOW,    "Hedging or uncertain language"
    else:
        level, rationale = ConfidenceLevel.UNKNOWN, "Very low confidence markers"

    return round(score, 4), level, rationale


def _contradicts(sent_a: str, sent_b: str) -> bool:
    """
    Heuristic: two sentences contradict if one negates a shared key term
    and the other asserts it.
    """
    tokens_a = set(re.findall(r"\b\w+\b", sent_a.lower()))
    tokens_b = set(re.findall(r"\b\w+\b", sent_b.lower()))
    overlap  = tokens_a & tokens_b - {"the", "a", "an", "is", "it", "of", "in"}
    if len(overlap) < 3:
        return False
    neg_a = bool(_NEGATION_RE.search(sent_a))
    neg_b = bool(_NEGATION_RE.search(sent_b))
    return neg_a != neg_b  # exactly one negates → potential contradiction


def _severity(score_a: float, score_b: float) -> str:
    avg = (score_a + score_b) / 2
    if avg >= 0.65:
        return "high"
    if avg >= 0.35:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# CritiqueAgent
# ---------------------------------------------------------------------------

class CritiqueAgent(BaseAgent[SharedContext, CritiqueResult]):
    """
    Evaluates content quality at the claim level.

    Strategy:
      1. Extract candidate claims from RetrievalResult chunks (or raw documents).
      2. Score each claim with a confidence model (heuristic by default).
      3. Detect pairwise contradictions among high/medium-confidence claims.
      4. Compute an aggregate overall_quality score.

    Override _score_claim() or _detect_contradictions() to plug in an
    LLM-backed evaluator without changing contracts.
    """

    TIMEOUT_SECONDS = 45.0

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def validate_input(self, context: SharedContext) -> None:
        super().validate_input(context)
        if not context.query.strip():
            raise AgentValidationError(self.agent_name, "SharedContext.query must not be empty")

    def validate_output(self, result: CritiqueResult) -> None:
        super().validate_output(result)
        if not result.claim_scores:
            raise AgentValidationError(self.agent_name, "CritiqueResult.claim_scores must not be empty")

    # ------------------------------------------------------------------ #
    #  Core run                                                            #
    # ------------------------------------------------------------------ #

    def run(self, context: SharedContext) -> CritiqueResult:
        """
        Score claims and detect contradictions.

        Args:
            context: SharedContext with .query and optionally
                     agent_outputs['RetrievalAgent']

        Returns:
            CritiqueResult

        Raises:
            AgentExecutionError: on claim extraction or scoring failure
        """
        logger.info("Starting critique | run_id=%s", context.run_id)

        max_claims:  int   = int(context.metadata.get("max_claims", 20))
        min_thresh:  float = float(context.metadata.get("min_confidence_threshold", 0.3))

        try:
            sentences = self._collect_sentences(context, max_claims)
            if not sentences:
                sentences = [context.query]

            claim_scores   = self._score_claims(sentences)
            contradictions = self._detect_contradictions(claim_scores)
            overall        = self._aggregate_quality(claim_scores, contradictions)

        except AgentExecutionError:
            raise
        except Exception as exc:
            raise AgentExecutionError(self.agent_name, f"Critique failed: {exc}") from exc

        result = CritiqueResult(
            claim_scores=claim_scores,
            contradictions=contradictions,
            overall_quality=round(overall, 4),
            metadata={
                "run_id":            context.run_id,
                "claims_evaluated":  len(claim_scores),
                "contradictions":    len(contradictions),
                "min_threshold":     min_thresh,
            },
        )
        logger.info(
            "Critique complete | claims=%d contradictions=%d quality=%.3f",
            len(claim_scores), len(contradictions), overall,
        )
        context.store_agent_output(self.agent_name, result)
        return result

    # ------------------------------------------------------------------ #
    #  Sentence collection from context                                   #
    # ------------------------------------------------------------------ #

    def _collect_sentences(self, context: SharedContext, max_claims: int) -> List[str]:
        """Gather candidate claim sentences from retrieval output or raw docs."""
        texts: List[str] = []

        # Prefer already-retrieved chunks
        retrieval: Optional[RetrievalResult] = context.get_agent_output("RetrievalAgent")
        if retrieval and retrieval.chunks:
            for chunk in retrieval.chunks:
                texts.append(chunk.content)
        elif context.documents:
            for doc in context.documents:
                texts.append(doc.content)
        else:
            texts.append(context.query)

        sentences: List[str] = []
        for text in texts:
            sentences.extend(_extract_sentences(text))

        return sentences[:max_claims]

    # ------------------------------------------------------------------ #
    #  Override-friendly scoring                                          #
    # ------------------------------------------------------------------ #

    def _score_claims(self, sentences: List[str]) -> List[ClaimScore]:
        """
        Score each sentence as a claim.

        Override to integrate an LLM evaluator.
        """
        scores: List[ClaimScore] = []
        for i, sent in enumerate(sentences):
            score_val, level, rationale = self._score_claim(sent)
            scores.append(
                ClaimScore(
                    claim_id=f"claim_{i:04d}_{uuid.uuid4().hex[:6]}",
                    claim_text=sent,
                    confidence=level,
                    score=score_val,
                    rationale=rationale,
                )
            )
        return scores

    def _score_claim(self, sentence: str) -> Tuple[float, ConfidenceLevel, str]:
        """
        Score a single sentence.

        Override this method to use an LLM or NLI model.

        Returns:
            (score 0–1, ConfidenceLevel, rationale)
        """
        return _confidence_score(sentence)

    # ------------------------------------------------------------------ #
    #  Override-friendly contradiction detection                          #
    # ------------------------------------------------------------------ #

    def _detect_contradictions(
        self, claim_scores: List[ClaimScore]
    ) -> List[Contradiction]:
        """
        Detect pairwise contradictions among claims.

        Override to use an NLI model or LLM.
        """
        contradictions: List[Contradiction] = []
        for cs_a, cs_b in itertools.combinations(claim_scores, 2):
            if _contradicts(cs_a.claim_text, cs_b.claim_text):
                sev = _severity(cs_a.score, cs_b.score)
                contradictions.append(
                    Contradiction(
                        contradiction_id=f"contra_{uuid.uuid4().hex[:8]}",
                        claim_a_id=cs_a.claim_id,
                        claim_b_id=cs_b.claim_id,
                        description=(
                            f"Potential contradiction between claims: "
                            f"'{cs_a.claim_text[:60]}…' vs '{cs_b.claim_text[:60]}…'"
                        ),
                        severity=sev,
                        resolution_hint="Prefer claim with higher confidence score or more recent source",
                    )
                )
        return contradictions

    # ------------------------------------------------------------------ #
    #  Aggregate quality                                                  #
    # ------------------------------------------------------------------ #

    def _aggregate_quality(
        self,
        claim_scores: List[ClaimScore],
        contradictions: List[Contradiction],
    ) -> float:
        if not claim_scores:
            return 0.0
        avg_score  = sum(c.score for c in claim_scores) / len(claim_scores)
        high_contra = sum(1 for c in contradictions if c.severity == "high")
        penalty    = high_contra * 0.05
        return max(0.0, min(1.0, avg_score - penalty))


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
    print("CritiqueAgent — debug run")
    print("=" * 60)

    ctx = SharedContext(query="Transformer attention mechanisms and their efficiency")
    ctx.add_document(
        doc_id="d1",
        source="paper_a",
        content=(
            "The transformer model demonstrates efficient attention computation. "
            "Multi-head attention is not efficient for very long sequences. "
            "FlashAttention confirms faster exact attention is possible."
        ),
    )
    ctx.add_document(
        doc_id="d2",
        source="paper_b",
        content=(
            "Sparse attention may reduce memory usage significantly. "
            "Some researchers suggest linear attention is not as accurate."
        ),
    )

    agent  = CritiqueAgent()
    result = agent(ctx)

    print("\n--- Claim Scores ---")
    for cs in result.claim_scores:
        print(f"  [{cs.claim_id}] {cs.confidence.name} ({cs.score:.3f}) — {cs.claim_text[:70]}")
        print(f"         rationale: {cs.rationale}")

    print("\n--- Contradictions ---")
    if result.contradictions:
        for c in result.contradictions:
            print(f"  [{c.contradiction_id}] severity={c.severity}")
            print(f"    {c.description}")
            print(f"    hint: {c.resolution_hint}")
    else:
        print("  None detected")

    print(f"\n--- Overall Quality: {result.overall_quality:.3f} ---")
    print("\n--- Metadata ---")
    print(json.dumps(result.metadata, indent=2))
    print("\nSchema validation: PASSED ✓")
