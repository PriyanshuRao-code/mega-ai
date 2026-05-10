"""
debug/run_agent_debug.py
=========================
End-to-end debug harness for the multi-agent pipeline.

What this file does:
  ① Tests every agent in isolation with a fresh SharedContext
  ② Tests agents in full pipeline order (decompose → retrieve → critique
     → synthesize → compress), sharing state via SharedContext
  ③ Validates every output schema via dataclass __post_init__ invariants
  ④ Prints detailed execution traces with timing and field summaries
  ⑤ Exercises error-path validation (bad inputs → expected exceptions)

Usage:
    cd <project_root>
    python debug/run_agent_debug.py

Exit codes:
    0  — all tests passed
    1  — one or more tests failed
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── make sure project root is on sys.path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── contracts + interfaces ─────────────────────────────────────────────────
from contracts.shared_context import SharedContext
from contracts.agent_contracts import (
    ClaimScore,
    ConfidenceLevel,
    Contradiction,
    DecompositionResult,
    RetrievalResult,
    RetrievedChunk,
    CritiqueResult,
    SynthesisResult,
    CompressionResult,
)
from interfaces.base_agent import AgentValidationError, AgentExecutionError

# ── agents ─────────────────────────────────────────────────────────────────
from agents.decomposition_agent import DecompositionAgent
from agents.retrieval_agent      import RetrievalAgent
from agents.critique_agent       import CritiqueAgent
from agents.synthesis_agent      import SynthesisAgent
from agents.compression_agent    import CompressionAgent


# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,          # suppress agent debug noise during tests
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("debug_runner")


# ===========================================================================
# Test helpers
# ===========================================================================

_PASS = "✓ PASS"
_FAIL = "✗ FAIL"

_results: List[Dict[str, Any]] = []


def _banner(title: str) -> None:
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _section(title: str) -> None:
    print(f"\n  ┌─ {title}")


def _run_test(
    name: str,
    fn: Callable[[], Any],
    expect_exception: Optional[type] = None,
) -> bool:
    """
    Execute fn() and record pass/fail.

    Args:
        name             : human-readable test label
        fn               : callable that runs the test
        expect_exception : if set, test passes only if this exception is raised
    """
    start = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - start
        if expect_exception:
            status = _FAIL
            note   = f"Expected {expect_exception.__name__} but no exception raised"
            passed = False
        else:
            status = _PASS
            note   = f"elapsed={elapsed:.3f}s"
            passed = True
    except Exception as exc:
        elapsed = time.perf_counter() - start
        if expect_exception and isinstance(exc, expect_exception):
            status = _PASS
            note   = f"Correctly raised {type(exc).__name__}: {exc}  elapsed={elapsed:.3f}s"
            passed = True
        else:
            status = _FAIL
            note   = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
            passed = False

    marker = "  │  "
    print(f"{marker}{status}  {name}")
    for line in note.splitlines():
        print(f"{marker}       {line}")
    _results.append({"name": name, "passed": passed, "note": note})
    return passed


def _pretty(obj: Any, max_len: int = 120) -> str:
    try:
        s = json.dumps(asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj,
                       indent=2, default=str)
    except Exception:
        s = str(obj)
    return s[:max_len] + ("…" if len(s) > max_len else "")


# ===========================================================================
# Shared fixture
# ===========================================================================

def _build_rich_context() -> SharedContext:
    """Return a SharedContext pre-populated with documents and history."""
    ctx = SharedContext(
        query=(
            "Find recent papers on transformer attention mechanisms, "
            "compare their efficiency claims, and synthesize a summary."
        ),
        metadata={
            "max_subtasks":          6,
            "max_hops":              2,
            "top_k":                 4,
            "min_score":             0.05,
            "max_claims":            15,
            "target_ratio":          0.45,
            "filler_aggressiveness": 0.8,
            "resolution_strategy":   "highest_confidence",
        },
    )
    ctx.add_message("user",      "Great question! Happy to help.")
    ctx.add_message("assistant", "Of course — let me look that up.")
    ctx.add_message("user",      ctx.query)

    ctx.add_document(
        doc_id="doc_001",
        source="https://arxiv.org/abs/1706.03762",
        content=(
            "Attention is all you need. The transformer model demonstrates "
            "efficient computation through multi-head self-attention. "
            "See [Vaswani et al.](https://arxiv.org/abs/1706.03762)."
        ),
    )
    ctx.add_document(
        doc_id="doc_002",
        source="https://arxiv.org/abs/2205.14135",
        content=(
            "FlashAttention is not just faster but also more memory-efficient. "
            "IO-aware exact attention confirms that speed and accuracy coexist. "
            "```python\n"
            "def flash_attention(Q, K, V): ...\n"
            "```\n"
            "| Method     | Complexity |\n"
            "| Standard   | O(n²)      |\n"
            "| Flash      | O(n²) IO   |\n"
        ),
    )
    ctx.add_document(
        doc_id="doc_003",
        source="https://arxiv.org/abs/2009.14794",
        content=(
            "Sparse attention may not always be as accurate as full attention. "
            "Linear attention is efficient but may lose expressiveness. "
            "In conclusion, no single method dominates all settings."
        ),
    )
    return ctx


# ===========================================================================
# ① Individual agent tests
# ===========================================================================

def test_decomposition_agent() -> None:
    _banner("① DecompositionAgent — Isolated Test")
    agent = DecompositionAgent()
    ctx   = _build_rich_context()

    _section("Happy-path: full decomposition")
    def happy():
        result = agent(ctx)
        assert isinstance(result, DecompositionResult), "Wrong return type"
        assert result.subtasks,        "subtasks must not be empty"
        assert result.execution_order, "execution_order must not be empty"
        assert len(result.execution_order) == len(result.subtasks), \
            "execution_order length must match subtasks"
        print(f"  │       subtasks={len(result.subtasks)}  order={result.execution_order}")
        for st in result.subtasks:
            print(f"  │         [{st.task_id}] ({st.category}) {st.description[:55]}")
    _run_test("DecompositionAgent happy-path", happy)

    _section("Schema validation: stored in context")
    def stored():
        stored_result = ctx.get_agent_output("DecompositionAgent")
        assert stored_result is not None, "Result not stored in context"
        assert isinstance(stored_result, DecompositionResult)
    _run_test("DecompositionAgent stores result in context", stored)

    _section("Error-path: empty query")
    def bad_query():
        bad_ctx = SharedContext(query="valid init")
        bad_ctx.query = "   "   # bypass __post_init__
        agent(bad_ctx)
    _run_test("DecompositionAgent rejects empty query", bad_query, AgentValidationError)

    _section("Dependency graph: retrieval precedes synthesis")
    def dep_order():
        r = ctx.get_agent_output("DecompositionAgent")
        if r and r.dependency_edges:
            for edge in r.dependency_edges:
                assert edge.source_task_id != edge.target_task_id, "Self-loop detected"
    _run_test("DecompositionAgent: no self-loops in edges", dep_order)


def test_retrieval_agent() -> None:
    _banner("② RetrievalAgent — Isolated Test")
    agent = RetrievalAgent()
    ctx   = _build_rich_context()

    _section("Happy-path: multi-hop retrieval")
    def happy():
        result = agent(ctx)
        assert isinstance(result, RetrievalResult), "Wrong return type"
        assert result.chunks, "chunks must not be empty"
        for chunk in result.chunks:
            assert 0.0 <= chunk.score <= 1.0, f"Invalid score: {chunk.score}"
            assert chunk.hop >= 0,            f"Invalid hop: {chunk.hop}"
        print(f"  │       total_chunks={len(result.chunks)}  total_hops={result.total_hops}")
        for c in result.chunks[:3]:
            print(f"  │         [{c.chunk_id}] hop={c.hop} score={c.score} src={c.source}")
    _run_test("RetrievalAgent happy-path", happy)

    _section("Provenance: every chunk has a provenance entry")
    def provenance():
        r = ctx.get_agent_output("RetrievalAgent")
        if r:
            chunk_ids = {c.chunk_id for c in r.chunks}
            prov_ids  = {p.claim_id for p in r.provenance_map}
            missing   = chunk_ids - prov_ids
            assert not missing, f"Chunks without provenance: {missing}"
    _run_test("RetrievalAgent provenance completeness", provenance)

    _section("Empty corpus: stub result")
    def empty_corpus():
        empty_ctx = SharedContext(query="test empty corpus")
        result    = agent(empty_ctx)
        assert result.chunks, "Should return stub chunk"
        assert result.metadata.get("stub"), "Should flag stub in metadata"
    _run_test("RetrievalAgent handles empty corpus gracefully", empty_corpus)

    _section("Citation tracking")
    def citations():
        r = ctx.get_agent_output("RetrievalAgent")
        if r:
            chunks_with_citations = [c for c in r.chunks if c.citations]
            print(f"  │       chunks_with_citations={len(chunks_with_citations)}")
    _run_test("RetrievalAgent citation tracking runs without error", citations)


def test_critique_agent() -> None:
    _banner("③ CritiqueAgent — Isolated Test")
    agent = CritiqueAgent()
    ctx   = _build_rich_context()
    # Pre-populate RetrievalAgent output
    RetrievalAgent()(ctx)

    _section("Happy-path: claim scoring")
    def happy():
        result = agent(ctx)
        assert isinstance(result, CritiqueResult), "Wrong return type"
        assert result.claim_scores, "claim_scores must not be empty"
        assert 0.0 <= result.overall_quality <= 1.0
        print(f"  │       claims={len(result.claim_scores)}  contradictions={len(result.contradictions)}")
        print(f"  │       overall_quality={result.overall_quality:.3f}")
        for cs in result.claim_scores[:3]:
            print(f"  │         [{cs.claim_id}] {cs.confidence.name} ({cs.score:.3f}) {cs.claim_text[:50]}")
    _run_test("CritiqueAgent happy-path", happy)

    _section("Contradiction: severity values valid")
    def contra_severity():
        r = ctx.get_agent_output("CritiqueAgent")
        if r:
            for c in r.contradictions:
                assert c.severity in {"high", "medium", "low"}, \
                    f"Invalid severity: {c.severity}"
    _run_test("CritiqueAgent contradiction severity values valid", contra_severity)

    _section("Fallback: no retrieval output in context")
    def no_retrieval():
        bare_ctx = SharedContext(query="test without retrieval")
        bare_ctx.add_document(doc_id="d1", source="s1", content="Attention is efficient.")
        result = CritiqueAgent()(bare_ctx)
        assert result.claim_scores
    _run_test("CritiqueAgent works without prior RetrievalAgent output", no_retrieval)


def test_synthesis_agent() -> None:
    _banner("④ SynthesisAgent — Isolated Test")
    agent = SynthesisAgent()
    ctx   = _build_rich_context()
    # Pre-populate pipeline outputs
    RetrievalAgent()(ctx)
    CritiqueAgent()(ctx)

    _section("Happy-path: merge + resolution")
    def happy():
        result = agent(ctx)
        assert isinstance(result, SynthesisResult), "Wrong return type"
        assert result.merged_output.strip(), "merged_output must not be empty"
        assert 0.0 <= result.confidence <= 1.0
        print(f"  │       confidence={result.confidence:.3f}")
        print(f"  │       resolved_contradictions={len(result.resolved_contradictions)}")
        print(f"  │       contributing_sources={result.contributing_sources}")
        print(f"  │       merged_output[:120]:")
        for line in result.merged_output[:240].splitlines():
            print(f"  │         {line}")
    _run_test("SynthesisAgent happy-path", happy)

    _section("Contradiction resolution logged")
    def resolution_logged():
        r = ctx.get_agent_output("SynthesisAgent")
        if r:
            for res in r.resolved_contradictions:
                assert res.resolution, "resolution field must not be empty"
                assert res.rationale,  "rationale field must not be empty"
    _run_test("SynthesisAgent resolution fields populated", resolution_logged)

    _section("Error-path: no usable content")
    def no_content():
        empty_ctx = SharedContext(query="no docs no outputs")
        SynthesisAgent()(empty_ctx)
    _run_test("SynthesisAgent raises error with no content", no_content, AgentExecutionError)


def test_compression_agent() -> None:
    _banner("⑤ CompressionAgent — Isolated Test")
    agent = CompressionAgent()
    ctx   = _build_rich_context()
    # Pre-populate full pipeline
    RetrievalAgent()(ctx)
    CritiqueAgent()(ctx)
    SynthesisAgent()(ctx)

    _section("Happy-path: compress full pipeline output")
    def happy():
        result = agent(ctx)
        assert isinstance(result, CompressionResult), "Wrong return type"
        assert result.compressed_text.strip()
        assert result.compression_ratio > 0.0
        assert result.tokens_saved >= 0
        print(f"  │       ratio={result.compression_ratio:.3f}  saved={result.tokens_saved} tokens")
        print(f"  │       structured_blocks={len(result.structured_blocks)}")
        for blk in result.structured_blocks:
            print(f"  │         [{blk.block_id}] type={blk.block_type} pos={blk.position_hint}")
    _run_test("CompressionAgent happy-path", happy)

    _section("Lossless: all structured blocks preserved verbatim")
    def lossless():
        r = ctx.get_agent_output("CompressionAgent")
        if r:
            for blk in r.structured_blocks:
                assert blk.original.strip(), f"Block {blk.block_id} original is empty"
    _run_test("CompressionAgent structured blocks non-empty", lossless)

    _section("Compression ratio in valid range")
    def ratio_valid():
        r = ctx.get_agent_output("CompressionAgent")
        if r:
            assert 0 < r.compression_ratio <= 2.0, \
                f"Suspicious compression_ratio: {r.compression_ratio}"
    _run_test("CompressionAgent compression_ratio valid", ratio_valid)


# ===========================================================================
# ② Full pipeline integration test
# ===========================================================================

def test_full_pipeline() -> None:
    _banner("⑥ Full Pipeline Integration Test")

    ctx = _build_rich_context()
    pipeline = [
        ("DecompositionAgent", DecompositionAgent()),
        ("RetrievalAgent",     RetrievalAgent()),
        ("CritiqueAgent",      CritiqueAgent()),
        ("SynthesisAgent",     SynthesisAgent()),
        ("CompressionAgent",   CompressionAgent()),
    ]

    print(f"\n  Context run_id : {ctx.run_id}")
    print(f"  Query          : {ctx.query[:80]}…")

    for agent_name, agent in pipeline:
        def _step(a=agent, n=agent_name):
            t0     = time.perf_counter()
            result = a(ctx)
            elapsed = time.perf_counter() - t0
            stored  = ctx.get_agent_output(n)
            assert stored is not None, f"{n} did not store output in context"
            print(f"  │       elapsed={elapsed:.3f}s  result_type={type(result).__name__}")

        _run_test(f"Pipeline step: {agent_name}", _step)

    # Final assertion: all agents stored their outputs
    def all_stored():
        for name, _ in pipeline:
            out = ctx.get_agent_output(name)
            assert out is not None, f"{name} output missing from context"
        keys = list(ctx.agent_outputs.keys())
        print(f"  │       stored_outputs={keys}")
    _run_test("All agent outputs stored in SharedContext", all_stored)


# ===========================================================================
# ③ Schema validation stress tests
# ===========================================================================

def test_schema_validation() -> None:
    _banner("⑦ Contract / Schema Validation Tests")

    from contracts.agent_contracts import (
        ContractValidationError,
        SubTask, TaskStatus,
        DecompositionResult,
        RetrievedChunk,
        RetrievalResult,
        ClaimScore, ConfidenceLevel,
        CritiqueResult,
        SynthesisResult,
        CompressionResult,
        StructuredBlock,
    )
    from contracts.shared_context import SharedContextValidationError

    def test_empty_query():
        SharedContext(query="")
    _run_test("SharedContext rejects empty query", test_empty_query, SharedContextValidationError)

    def test_invalid_message_role():
        from contracts.shared_context import Message
        Message(role="robot", content="hi")
    _run_test("Message rejects invalid role", test_invalid_message_role, SharedContextValidationError)

    def test_decomp_empty_subtasks():
        DecompositionResult(subtasks=[])
    _run_test("DecompositionResult rejects empty subtasks", test_decomp_empty_subtasks, ContractValidationError)

    def test_chunk_bad_score():
        RetrievedChunk(chunk_id="c1", doc_id="d1", source="s", content="x", score=1.5)
    _run_test("RetrievedChunk rejects score > 1", test_chunk_bad_score, ContractValidationError)

    def test_retrieval_empty_chunks():
        RetrievalResult(chunks=[])
    _run_test("RetrievalResult rejects empty chunks", test_retrieval_empty_chunks, ContractValidationError)

    def test_claim_bad_score():
        ClaimScore(claim_id="c", claim_text="x", confidence=ConfidenceLevel.HIGH, score=-0.1)
    _run_test("ClaimScore rejects negative score", test_claim_bad_score, ContractValidationError)

    def test_critique_empty_claims():
        CritiqueResult(claim_scores=[], overall_quality=0.5)
    _run_test("CritiqueResult rejects empty claim_scores", test_critique_empty_claims, ContractValidationError)

    def test_critique_bad_quality():
        cs = ClaimScore(claim_id="x", claim_text="y", confidence=ConfidenceLevel.LOW, score=0.5)
        CritiqueResult(claim_scores=[cs], overall_quality=1.5)
    _run_test("CritiqueResult rejects quality > 1", test_critique_bad_quality, ContractValidationError)

    def test_synthesis_empty_output():
        SynthesisResult(merged_output="   ", confidence=0.5)
    _run_test("SynthesisResult rejects whitespace-only output", test_synthesis_empty_output, ContractValidationError)

    def test_compression_empty_text():
        CompressionResult(compressed_text="", compression_ratio=0.5)
    _run_test("CompressionResult rejects empty text", test_compression_empty_text, ContractValidationError)

    def test_compression_zero_ratio():
        CompressionResult(compressed_text="ok", compression_ratio=0.0)
    _run_test("CompressionResult rejects zero ratio", test_compression_zero_ratio, ContractValidationError)

    def test_contradiction_bad_severity():
        from contracts.agent_contracts import Contradiction
        Contradiction(
            contradiction_id="x", claim_a_id="a", claim_b_id="b",
            description="test", severity="extreme"
        )
    _run_test("Contradiction rejects invalid severity", test_contradiction_bad_severity, ContractValidationError)


# ===========================================================================
# ④ Context snapshot
# ===========================================================================

def print_context_snapshot(ctx: SharedContext) -> None:
    _banner("SharedContext Final Snapshot")
    snap = ctx.summary()
    for k, v in snap.items():
        print(f"  {k:30s}: {v}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    print("\n" + "╔" + "═" * 62 + "╗")
    print("║  Multi-Agent Pipeline — Full Debug & Validation Suite     ║")
    print("╚" + "═" * 62 + "╝")

    test_decomposition_agent()
    test_retrieval_agent()
    test_critique_agent()
    test_synthesis_agent()
    test_compression_agent()
    test_full_pipeline()
    test_schema_validation()

    # Build the final shared context for snapshot
    ctx = _build_rich_context()
    for AgentClass in [DecompositionAgent, RetrievalAgent, CritiqueAgent,
                       SynthesisAgent, CompressionAgent]:
        try:
            AgentClass()(ctx)
        except Exception:
            pass
    print_context_snapshot(ctx)

    # ── Summary ────────────────────────────────────────────────────────
    total  = len(_results)
    passed = sum(1 for r in _results if r["passed"])
    failed = total - passed

    _banner("Test Summary")
    print(f"  Total  : {total}")
    print(f"  Passed : {passed}  ✓")
    print(f"  Failed : {failed}  {'✗' if failed else '—'}")

    if failed:
        print("\n  FAILED TESTS:")
        for r in _results:
            if not r["passed"]:
                print(f"    ✗ {r['name']}")
                for line in r["note"].splitlines():
                    print(f"        {line}")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
