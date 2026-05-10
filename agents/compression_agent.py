"""
agents/compression_agent.py
============================
CompressionAgent — losslessly preserves structured data blocks (tables,
code, JSON, lists, formulas) while compressing conversational filler and
verbose narrative prose.

SOLID Alignment:
  - (S) Responsible only for context compression; no retrieval or evaluation
  - (O) Override _summarize_prose() to swap in an LLM summarizer
  - (L) Safe wherever BaseAgent[SharedContext, CompressionResult] is expected
  - (D) Depends only on interfaces/contracts; agnostic of other agents

Imports (external):
  stdlib  : logging, re, uuid
  local   : interfaces.base_agent, contracts.shared_context,
            contracts.agent_contracts

Input:
  SharedContext
    .query               — user query (required)
    .conversation_history — messages to compress (primary target)
    .documents            — documents to compress (secondary target)
    .agent_outputs        — optionally reads 'SynthesisAgent' output as source
    .metadata             — optional hints:
                             'target_ratio'       (float, default 0.5)
                             'preserve_last_n'    (int,   default 2)
                             'filler_aggressiveness' (float 0–1, default 0.7)

Output:
  CompressionResult
    .compressed_text   — compressed prose + placeholder markers for blocks
    .structured_blocks — losslessly preserved structured data
    .compression_ratio — len(output) / len(input)
    .tokens_saved      — approximate token reduction
    .metadata          — diagnostics

Exceptions:
  AgentValidationError   : query empty or no text content found
  AgentExecutionError    : extraction or compression fails
  ContractValidationError: output invariant violation

Dependencies:
  interfaces.base_agent.BaseAgent
  contracts.shared_context.SharedContext
  contracts.agent_contracts.CompressionResult, StructuredBlock, SynthesisResult
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Dict, List, Optional, Tuple

from interfaces.base_agent import BaseAgent, AgentExecutionError, AgentValidationError
from contracts.shared_context import SharedContext
from contracts.agent_contracts import (
    CompressionResult,
    ContractValidationError,
    StructuredBlock,
    SynthesisResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured block patterns
# ---------------------------------------------------------------------------

# Each entry: (block_type, compiled_regex)
# Patterns designed to match common structured regions verbatim
_BLOCK_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # Fenced code blocks (```...```)
    ("code",    re.compile(r"```[\s\S]*?```", re.MULTILINE)),
    # Inline code spans
    ("code",    re.compile(r"`[^`\n]{3,}`")),
    # Markdown tables (rows with |)
    ("table",   re.compile(r"(?:^\|.+\|\n)+", re.MULTILINE)),
    # JSON-like objects/arrays
    ("json",    re.compile(r"\{[\s\S]{20,}?\}|\[[\s\S]{20,}?\]")),
    # Numbered or bulleted lists (3+ items)
    ("list",    re.compile(r"(?:^[ \t]*(?:\d+\.|[-*+])[ \t]+.+\n?){3,}", re.MULTILINE)),
    # Mathematical/formula lines
    ("formula", re.compile(r"(?:\$\$[\s\S]+?\$\$|\$[^$\n]+\$)")),
]

# Conversational filler patterns (phrases to strip)
_FILLER_RE = re.compile(
    r"\b("
    r"of course|certainly|sure|absolutely|definitely|"
    r"great question|good question|happy to help|glad to|"
    r"feel free to|don't hesitate|please note that|it is worth noting that|"
    r"it's important to note that|as (I|we) mentioned|as mentioned above|"
    r"in conclusion|to summarize|in summary|let me explain|"
    r"as you (may|can) (see|know)|needless to say"
    r")\b[,.]?",
    re.I,
)

# Approximate tokens ≈ words * 1.3
def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# CompressionAgent
# ---------------------------------------------------------------------------

class CompressionAgent(BaseAgent[SharedContext, CompressionResult]):
    """
    Compresses context while preserving structured data losslessly.

    Pipeline:
      1. Collect raw text from conversation history, documents, or SynthesisResult.
      2. Extract structured blocks (code, tables, JSON, lists, formulas).
      3. Replace extracted blocks with placeholder tokens in the text.
      4. Strip conversational filler from the remaining prose.
      5. Summarize verbose prose sections (heuristic sentence scoring).
      6. Reinsert placeholders in compressed_text (blocks stored separately).

    Override _summarize_prose() to integrate an LLM-based summarizer.
    """

    TIMEOUT_SECONDS = 30.0

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def validate_input(self, context: SharedContext) -> None:
        super().validate_input(context)
        if not context.query.strip():
            raise AgentValidationError(self.agent_name, "SharedContext.query must not be empty")

    def validate_output(self, result: CompressionResult) -> None:
        super().validate_output(result)
        if not result.compressed_text.strip():
            raise AgentValidationError(
                self.agent_name, "CompressionResult.compressed_text must not be empty"
            )

    # ------------------------------------------------------------------ #
    #  Core run                                                            #
    # ------------------------------------------------------------------ #

    def run(self, context: SharedContext) -> CompressionResult:
        """
        Compress context content.

        Args:
            context: SharedContext

        Returns:
            CompressionResult

        Raises:
            AgentExecutionError: if compression fails
        """
        logger.info("Starting compression | run_id=%s", context.run_id)

        target_ratio:          float = float(context.metadata.get("target_ratio", 0.5))
        preserve_last_n:       int   = int(context.metadata.get("preserve_last_n", 2))
        filler_aggressiveness: float = float(context.metadata.get("filler_aggressiveness", 0.7))

        try:
            raw_text = self._collect_raw_text(context, preserve_last_n)
            if not raw_text.strip():
                raise AgentExecutionError(
                    self.agent_name, "No text content found to compress"
                )

            original_tokens = _approx_tokens(raw_text)

            # Step 1: Extract structured blocks losslessly
            stripped_text, blocks = self._extract_structured_blocks(raw_text)

            # Step 2: Strip conversational filler
            stripped_text = self._strip_filler(stripped_text, filler_aggressiveness)

            # Step 3: Summarize remaining prose
            compressed_prose = self._summarize_prose(stripped_text, target_ratio)

            if not compressed_prose.strip():
                compressed_prose = context.query  # guaranteed non-empty fallback

            compressed_tokens = _approx_tokens(compressed_prose)
            ratio = round(
                len(compressed_prose) / max(len(raw_text), 1), 4
            )
            saved = max(0, original_tokens - compressed_tokens)

        except AgentExecutionError:
            raise
        except Exception as exc:
            raise AgentExecutionError(self.agent_name, f"Compression failed: {exc}") from exc

        result = CompressionResult(
            compressed_text=compressed_prose,
            structured_blocks=blocks,
            compression_ratio=max(ratio, 0.01),  # must be > 0
            tokens_saved=saved,
            metadata={
                "run_id":           context.run_id,
                "original_tokens":  original_tokens,
                "compressed_tokens":compressed_tokens,
                "blocks_preserved": len(blocks),
                "target_ratio":     target_ratio,
            },
        )
        logger.info(
            "Compression complete | ratio=%.3f saved_tokens=%d blocks=%d",
            ratio, saved, len(blocks),
        )
        context.store_agent_output(self.agent_name, result)
        return result

    # ------------------------------------------------------------------ #
    #  Text collection                                                     #
    # ------------------------------------------------------------------ #

    def _collect_raw_text(self, context: SharedContext, preserve_last_n: int) -> str:
        """
        Assemble raw text to compress.

        Priority:
          1. SynthesisAgent output (most processed)
          2. Conversation history (older turns only; recent turns are preserved)
          3. Document contents
          4. Query as last resort
        """
        parts: List[str] = []

        synthesis: Optional[SynthesisResult] = context.get_agent_output("SynthesisAgent")
        if synthesis:
            parts.append(synthesis.merged_output)

        history = context.conversation_history
        compressible_turns = history[: max(0, len(history) - preserve_last_n)]
        for msg in compressible_turns:
            parts.append(f"{msg.role}: {msg.content}")

        for doc in context.documents:
            parts.append(doc.content)

        if not parts:
            parts.append(context.query)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #  Structured block extraction (lossless)                            #
    # ------------------------------------------------------------------ #

    def _extract_structured_blocks(
        self, text: str
    ) -> Tuple[str, List[StructuredBlock]]:
        """
        Find and remove structured blocks from text.

        Blocks are stored verbatim in StructuredBlock objects; a placeholder
        token `[BLOCK:block_id]` is left in the text so position is tracked.

        Returns:
            (text with placeholders, list of StructuredBlock)
        """
        blocks: List[StructuredBlock] = []
        result_text = text

        for block_type, pattern in _BLOCK_PATTERNS:
            for match in list(pattern.finditer(result_text)):
                block_id = f"blk_{block_type}_{uuid.uuid4().hex[:8]}"
                original = match.group(0)
                if len(original.strip()) < 10:
                    continue  # skip trivially small matches
                blocks.append(
                    StructuredBlock(
                        block_id=block_id,
                        block_type=block_type,
                        original=original,
                        position_hint=match.start(),
                    )
                )
                result_text = result_text.replace(original, f"[BLOCK:{block_id}]", 1)

        logger.debug("Extracted %d structured blocks", len(blocks))
        return result_text, blocks

    # ------------------------------------------------------------------ #
    #  Filler removal                                                      #
    # ------------------------------------------------------------------ #

    def _strip_filler(self, text: str, aggressiveness: float) -> str:
        """
        Remove conversational filler phrases.

        aggressiveness controls how many filler occurrences are stripped:
          0.0 → none stripped
          1.0 → all stripped
        """
        if aggressiveness <= 0.0:
            return text
        matches = list(_FILLER_RE.finditer(text))
        threshold = int(len(matches) * aggressiveness)
        to_remove = matches[:threshold]
        for match in reversed(to_remove):
            text = text[: match.start()] + text[match.end():]
        return text.strip()

    # ------------------------------------------------------------------ #
    #  Override-friendly prose summarization                             #
    # ------------------------------------------------------------------ #

    def _summarize_prose(self, text: str, target_ratio: float) -> str:
        """
        Heuristic extractive summarization of prose text.

        Override this method to integrate an abstractive LLM summarizer.

        Strategy:
          - Score each sentence by: length, keyword density relative to
            the full text, and position (earlier = higher weight).
          - Keep top-N sentences where N ≈ target_ratio * total_sentences.

        Args:
            text         : prose with filler already removed
            target_ratio : target length fraction (0 < ratio <= 1)

        Returns:
            Compressed string (never empty)
        """
        sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return text[:500] if text else ""

        target_n = max(1, int(len(sentences) * target_ratio))

        # Score: prefer sentences with higher token density
        all_tokens = re.findall(r"\b\w+\b", text.lower())
        freq: Dict[str, int] = {}
        for t in all_tokens:
            freq[t] = freq.get(t, 0) + 1

        def _score(idx: int, sent: str) -> float:
            tokens = re.findall(r"\b\w+\b", sent.lower())
            if not tokens:
                return 0.0
            density = sum(freq.get(t, 0) for t in tokens) / len(tokens)
            position_bonus = 1.0 / (idx + 1)  # earlier sentences score higher
            return density + position_bonus

        scored = sorted(
            enumerate(sentences),
            key=lambda x: _score(x[0], x[1]),
            reverse=True,
        )
        kept_indices = sorted(idx for idx, _ in scored[:target_n])
        return " ".join(sentences[i] for i in kept_indices)


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
    print("CompressionAgent — debug run")
    print("=" * 60)

    ctx = SharedContext(
        query="Summarize transformer attention mechanisms",
        metadata={"target_ratio": 0.4, "preserve_last_n": 1, "filler_aggressiveness": 0.8},
    )

    ctx.add_message("user",      "Great question! Can you explain transformers?")
    ctx.add_message("assistant", "Of course! Happy to help. Transformers use self-attention.")
    ctx.add_message("user",      "Please summarize the key efficiency findings.")

    ctx.add_document(
        doc_id="d1", source="paper_a",
        content=(
            "Of course, transformers are powerful. Happy to elaborate. "
            "The attention mechanism computes pairwise interactions:\n\n"
            "```python\n"
            "def attention(Q, K, V):\n"
            "    scores = Q @ K.T / sqrt(d_k)\n"
            "    return softmax(scores) @ V\n"
            "```\n\n"
            "FlashAttention demonstrates IO-aware computation that is significantly faster. "
            "Sparse attention reduces complexity from O(n²) to O(n log n). "
            "The following table compares methods:\n\n"
            "| Method       | Complexity | Memory |\n"
            "| Full         | O(n²)      | High   |\n"
            "| Sparse       | O(n log n) | Medium |\n"
            "| FlashAttn    | O(n²)      | Low    |\n\n"
            "In conclusion, efficient attention remains an active research area."
        ),
    )

    agent  = CompressionAgent()
    result = agent(ctx)

    print("\n--- Compressed Text ---")
    print(result.compressed_text)

    print(f"\n--- Structured Blocks Preserved: {len(result.structured_blocks)} ---")
    for blk in result.structured_blocks:
        print(f"  [{blk.block_id}] type={blk.block_type} pos={blk.position_hint}")
        print(f"    original[:80]: {blk.original[:80].strip()!r}")

    print(f"\n--- Compression Ratio: {result.compression_ratio:.3f} ---")
    print(f"--- Tokens Saved: {result.tokens_saved} ---")
    print("\n--- Metadata ---")
    print(json.dumps(result.metadata, indent=2))
    print("\nSchema validation: PASSED ✓")
