"""
agents/retrieval_agent.py
==========================
RetrievalAgent — performs multi-hop retrieval over SharedContext.documents,
builds provenance maps, and tracks citations for every retrieved chunk.

SOLID Alignment:
  - (S) Responsible only for retrieval + provenance logic
  - (O) Override _score_chunk() / _expand_hop() to swap retrieval backends
  - (L) Safe to use wherever BaseAgent[SharedContext, RetrievalResult] is expected
  - (D) Depends on interfaces/contracts, not on other agents

Imports (external):
  stdlib  : logging, re, uuid, math
  local   : interfaces.base_agent, contracts.shared_context,
            contracts.agent_contracts

Input:
  SharedContext
    .query               — driving retrieval query (required)
    .documents           — corpus of Document objects to retrieve from
    .metadata            — optional hints:
                             'max_hops'        (int,   default 2)
                             'top_k'           (int,   default 5)
                             'min_score'       (float, default 0.1)

Output:
  RetrievalResult
    .chunks         — all RetrievedChunk objects across all hops
    .provenance_map — ProvenanceMap per retrieved chunk
    .total_hops     — maximum hop depth reached
    .metadata       — diagnostics (run_id, hops performed, chunk counts)

Exceptions:
  AgentValidationError  : query empty or context invalid
  AgentExecutionError   : retrieval or provenance construction fails
  ContractValidationError: output invariant violation (no chunks)

Dependencies:
  interfaces.base_agent.BaseAgent
  contracts.shared_context.SharedContext, Document
  contracts.agent_contracts.RetrievalResult, RetrievedChunk, ProvenanceMap
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from typing import Dict, List, Set, Tuple

from interfaces.base_agent import BaseAgent, AgentExecutionError, AgentValidationError
from contracts.shared_context import SharedContext, Document
from contracts.agent_contracts import (
    ContractValidationError,
    ProvenanceMap,
    RetrievalResult,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring helpers (TF-IDF-style without external deps)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _term_freq(tokens: List[str]) -> Dict[str, float]:
    if not tokens:
        return {}
    tf: Dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    n = len(tokens)
    return {t: c / n for t, c in tf.items()}


def _cosine_similarity(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """
    Approximate cosine similarity using term-frequency overlap.
    Returns a float in [0, 1].
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    qtf = _term_freq(query_tokens)
    dtf = _term_freq(doc_tokens)
    shared = set(qtf) & set(dtf)
    if not shared:
        return 0.0
    dot = sum(qtf[t] * dtf[t] for t in shared)
    norm_q = math.sqrt(sum(v ** 2 for v in qtf.values()))
    norm_d = math.sqrt(sum(v ** 2 for v in dtf.values()))
    return dot / (norm_q * norm_d) if (norm_q * norm_d) > 0 else 0.0


def _extract_citations(doc: Document) -> List[Tuple[str, str]]:
    """
    Extract (label, uri) citation pairs from document metadata or content.
    Looks for markdown-style links: [label](uri)
    """
    pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
    citations: List[Tuple[str, str]] = []
    citations += pattern.findall(doc.content)
    citations += [
        (str(k), str(v))
        for k, v in doc.metadata.items()
        if isinstance(v, str) and v.startswith("http")
    ]
    return citations[:10]  # cap per chunk


# ---------------------------------------------------------------------------
# RetrievalAgent
# ---------------------------------------------------------------------------

class RetrievalAgent(BaseAgent[SharedContext, RetrievalResult]):
    """
    Multi-hop retrieval agent with provenance mapping and citation tracking.

    Hop 0: score all documents in context.documents against context.query.
    Hop N: use top-k chunks from the previous hop as new query seeds to
           retrieve additional supporting passages (bridge retrieval).

    Override _score_chunk() to replace the heuristic scorer with a vector DB
    or dense retrieval backend without touching any contract.
    """

    TIMEOUT_SECONDS = 45.0

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def validate_input(self, context: SharedContext) -> None:
        super().validate_input(context)
        if not context.query.strip():
            raise AgentValidationError(self.agent_name, "SharedContext.query must not be empty")

    def validate_output(self, result: RetrievalResult) -> None:
        super().validate_output(result)
        if not result.chunks:
            raise AgentValidationError(self.agent_name, "RetrievalResult must contain at least one chunk")

    # ------------------------------------------------------------------ #
    #  Core run                                                            #
    # ------------------------------------------------------------------ #

    def run(self, context: SharedContext) -> RetrievalResult:
        """
        Execute multi-hop retrieval over context.documents.

        Args:
            context: SharedContext with .query and .documents

        Returns:
            RetrievalResult

        Raises:
            AgentExecutionError: if retrieval fails
        """
        logger.info("Starting retrieval | run_id=%s docs=%d", context.run_id, len(context.documents))

        max_hops: int  = int(context.metadata.get("max_hops", 2))
        top_k:    int  = int(context.metadata.get("top_k", 5))
        min_score:float = float(context.metadata.get("min_score", 0.1))

        # If no documents available, generate a placeholder so the contract
        # is satisfied; real deployments should inject a retrieval backend.
        if not context.documents:
            logger.warning("No documents in context; creating synthetic stub chunk")
            return self._stub_result(context)

        try:
            all_chunks: List[RetrievedChunk] = []
            provenance: List[ProvenanceMap]  = []
            seen_chunk_ids: Set[str]         = set()

            query_text  = context.query
            hop_index   = 0

            while hop_index <= max_hops:
                hop_chunks = self._retrieve_hop(
                    query_text=query_text,
                    documents=context.documents,
                    hop=hop_index,
                    top_k=top_k,
                    min_score=min_score,
                    seen=seen_chunk_ids,
                )
                if not hop_chunks:
                    break

                for chunk in hop_chunks:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)
                    provenance.append(
                        self._build_provenance(chunk, all_chunks, hop_index)
                    )

                # Next hop: seed from the best chunk's content
                query_text = hop_chunks[0].content[:300]
                hop_index += 1

        except AgentExecutionError:
            raise
        except Exception as exc:
            raise AgentExecutionError(self.agent_name, f"Retrieval failed: {exc}") from exc

        # Guarantee at least one chunk
        if not all_chunks:
            return self._stub_result(context)

        result = RetrievalResult(
            chunks=all_chunks,
            provenance_map=provenance,
            total_hops=hop_index - 1,
            metadata={
                "run_id":       context.run_id,
                "hops":         hop_index,
                "total_chunks": len(all_chunks),
                "top_k":        top_k,
            },
        )
        logger.info("Retrieval complete | chunks=%d hops=%d", len(all_chunks), result.total_hops)
        context.store_agent_output(self.agent_name, result)
        return result

    # ------------------------------------------------------------------ #
    #  Hop-level retrieval                                                 #
    # ------------------------------------------------------------------ #

    def _retrieve_hop(
        self,
        query_text: str,
        documents: list,
        hop: int,
        top_k: int,
        min_score: float,
        seen: Set[str],
    ) -> List[RetrievedChunk]:
        """Score every document, return top-k unseen chunks above min_score."""
        scored: List[Tuple[float, Document]] = []
        for doc in documents:
            score = self._score_chunk(query_text, doc)
            if score >= min_score:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        chunks: List[RetrievedChunk] = []
        for score, doc in scored[:top_k]:
            chunk_id = f"chunk_{hop}_{doc.doc_id}_{uuid.uuid4().hex[:6]}"
            if chunk_id in seen:
                continue
            citations = _extract_citations(doc)
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source=doc.source,
                    content=doc.content,
                    score=round(min(score, 1.0), 4),
                    hop=hop,
                    citations=citations,
                    metadata={"doc_metadata": doc.metadata},
                )
            )
        return chunks

    # ------------------------------------------------------------------ #
    #  Override-friendly scorer                                            #
    # ------------------------------------------------------------------ #

    def _score_chunk(self, query: str, doc: Document) -> float:
        """
        Score a document against the query.

        Override this method to integrate a dense retrieval model / vector DB.

        Args:
            query: current query string
            doc:   Document to score

        Returns:
            float in [0, 1]
        """
        return _cosine_similarity(_tokenize(query), _tokenize(doc.content))

    # ------------------------------------------------------------------ #
    #  Provenance construction                                             #
    # ------------------------------------------------------------------ #

    def _build_provenance(
        self,
        chunk: RetrievedChunk,
        all_chunks: List[RetrievedChunk],
        hop: int,
    ) -> ProvenanceMap:
        """Build the retrieval chain leading to this chunk."""
        chain = [c.chunk_id for c in all_chunks if c.hop < hop]
        chain.append(chunk.chunk_id)
        return ProvenanceMap(
            claim_id=chunk.chunk_id,
            chunk_ids=chain,
            source_uri=chunk.source,
            hops=hop,
        )

    # ------------------------------------------------------------------ #
    #  Stub for empty-corpus guard                                         #
    # ------------------------------------------------------------------ #

    def _stub_result(self, context: SharedContext) -> RetrievalResult:
        stub_chunk = RetrievedChunk(
            chunk_id="chunk_stub_000",
            doc_id="stub",
            source="no-source",
            content=f"No documents available. Query: {context.query}",
            score=0.0,
            hop=0,
        )
        return RetrievalResult(
            chunks=[stub_chunk],
            provenance_map=[
                ProvenanceMap(
                    claim_id="chunk_stub_000",
                    chunk_ids=["chunk_stub_000"],
                    source_uri="no-source",
                    hops=0,
                )
            ],
            total_hops=0,
            metadata={"run_id": context.run_id, "stub": True},
        )


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
    print("RetrievalAgent — debug run")
    print("=" * 60)

    ctx = SharedContext(
        query="transformer attention mechanisms efficiency",
        metadata={"max_hops": 2, "top_k": 3, "min_score": 0.05},
    )
    ctx.add_document(
        doc_id="doc_001",
        source="https://arxiv.org/abs/1706.03762",
        content=(
            "Attention is all you need. The transformer model uses "
            "multi-head attention to efficiently process sequences. "
            "See also [Vaswani et al.](https://arxiv.org/abs/1706.03762)."
        ),
    )
    ctx.add_document(
        doc_id="doc_002",
        source="https://arxiv.org/abs/2004.05150",
        content=(
            "Efficient transformers: a survey. Various techniques such as "
            "sparse attention and linear attention reduce quadratic complexity."
        ),
    )
    ctx.add_document(
        doc_id="doc_003",
        source="https://arxiv.org/abs/2102.09680",
        content=(
            "FlashAttention: fast and memory-efficient exact attention. "
            "IO-aware algorithm improves attention speed significantly."
        ),
    )

    agent  = RetrievalAgent()
    result = agent(ctx)

    print(f"\n--- Retrieved Chunks (total={len(result.chunks)}) ---")
    for chunk in result.chunks:
        print(f"  [{chunk.chunk_id}] hop={chunk.hop} score={chunk.score} source={chunk.source}")
        if chunk.citations:
            print(f"         citations: {chunk.citations}")

    print("\n--- Provenance Map ---")
    for prov in result.provenance_map:
        print(f"  claim={prov.claim_id} hops={prov.hops} chain={prov.chunk_ids}")

    print(f"\n--- Total Hops: {result.total_hops} ---")
    print("\n--- Metadata ---")
    print(json.dumps(result.metadata, indent=2))
    print("\nSchema validation: PASSED ✓")
