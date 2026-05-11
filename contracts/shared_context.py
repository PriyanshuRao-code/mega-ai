"""
contracts/shared_context.py
============================
Unified SharedContext for the entire lightweight multi-agent ecosystem.

This file is the SINGLE canonical shared state object used by:
    - agents
    - tools
    - pipelines
    - lightweight orchestration

Architecture Goals
------------------
- Dataclass-first (lightweight + easy debugging)
- Thread-safe for concurrent tool execution
- Extensible without breaking compatibility
- Shared globally across the entire project

SOLID Alignment
---------------
S — holds shared runtime state only
O — extend via optional fields; existing contracts remain stable
L — compatible with all generated agents/tools
I — no dependency on orchestration internals
D — stdlib-only lightweight dependency graph

Dependencies
------------
- stdlib only
"""

from __future__ import annotations

import threading
import uuid
from pydantic import BaseModel, Field, PrivateAttr, model_validator, ConfigDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ============================================================================
# Exceptions
# ============================================================================

class SharedContextValidationError(ValueError):
    """Raised when SharedContext receives invalid data."""


# ============================================================================
# Supporting Types
# ============================================================================

class Message(BaseModel):
    """
    Single conversational turn.
    """

    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_model(self):
        allowed_roles = {"user", "assistant", "system"}

        if self.role not in allowed_roles:
            raise SharedContextValidationError(
                f"Invalid role '{self.role}'. "
                f"Allowed: {sorted(allowed_roles)}"
            )

        if not isinstance(self.content, str):
            raise SharedContextValidationError(
                "Message.content must be a string."
            )


        return self
class Document(BaseModel):
    """
    Retrieved or injected document.
    """

    doc_id: str
    source: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    """
    Unified output container compatible with:
    - old tooling layer
    - reflection tools
    - generated agents
    - future pipeline execution

    Old tooling compatibility fields:
        step_id
        tool_name
        summary
        raw_data

    New agent compatibility fields:
        agent_name
        output
    """

    # ------------------------------------------------------------------
    # OLD TOOL CONTRACT FIELDS
    # ------------------------------------------------------------------

    step_id: str = ""
    tool_name: str = ""
    summary: str = ""
    raw_data: Any = None

    # ------------------------------------------------------------------
    # NEW AGENT CONTRACT FIELDS
    # ------------------------------------------------------------------

    agent_name: str = ""
    output: Any = None

    # ------------------------------------------------------------------
    # SHARED
    # ------------------------------------------------------------------

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Compatibility bridge
    # ------------------------------------------------------------------

    @model_validator(mode='after')
    def validate_model(self):

        # Tool → agent mapping
        if not self.agent_name and self.tool_name:
            self.agent_name = self.tool_name

        if self.output is None and self.raw_data is not None:
            self.output = self.raw_data

        # Agent → tool mapping
        if not self.tool_name and self.agent_name:
            self.tool_name = self.agent_name

        if self.raw_data is None and self.output is not None:
            self.raw_data = self.output


        return self
class TokenUsage(BaseModel):
    """
    Aggregated token accounting.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion


class ExecutionTraceEntry(BaseModel):
    """
    Lightweight observability event.
    """

    source: str
    step: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    duration_ms: Optional[float] = None
    notes: Optional[str] = None


# ============================================================================
# SharedContext
# ============================================================================

class SharedContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """
    Unified mutable runtime context shared across the entire pipeline.

    Backward compatible with:
    - generated tools layer
    - generated agents layer
    - future lightweight orchestration
    """

    # ----------------------------------------------------------------------
    # OLD TOOL-COMPATIBILITY FIELDS
    # ----------------------------------------------------------------------

    session_id: str = ""
    agent_id: str = ""

    # ----------------------------------------------------------------------
    # NEW GLOBAL FIELDS
    # ----------------------------------------------------------------------

    query: str = ""

    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ----------------------------------------------------------------------
    # Conversation
    # ----------------------------------------------------------------------

    conversation_history: List[Message] = Field(default_factory=list)

    # ----------------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------------

    documents: List[Document] = Field(default_factory=list)

    # ----------------------------------------------------------------------
    # Outputs
    # ----------------------------------------------------------------------

    agent_outputs: Dict[str, AgentOutput] = Field(default_factory=dict)

    tool_outputs: List[AgentOutput] = Field(default_factory=list)

    # ----------------------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------------------

    citations: List[str] = Field(default_factory=list)

    policy_violations: List[str] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    # ----------------------------------------------------------------------
    # Tokens
    # ----------------------------------------------------------------------

    token_usage: TokenUsage = Field(default_factory=TokenUsage)

    # ----------------------------------------------------------------------
    # Trace
    # ----------------------------------------------------------------------

    execution_trace: List[ExecutionTraceEntry] = Field(default_factory=list)

    # ----------------------------------------------------------------------
    # Internal KV Store
    # ----------------------------------------------------------------------

    _kv_store: Dict[str, Any] = PrivateAttr(default_factory=dict)

    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    # =========================================================================
    # Validation
    # =========================================================================

    @model_validator(mode='after')
    def validate_model(self):

        # --------------------------------------------------------------
        # TOOL-STYLE CONSTRUCTION
        # --------------------------------------------------------------

        if self.session_id and not self.query:
            self.query = f"session:{self.session_id}"

        # --------------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------------

        if not isinstance(self.query, str):
            raise SharedContextValidationError(
                "SharedContext.query must be a string."
            )

        if self.query == "":
            raise SharedContextValidationError(
                "SharedContext.query must not be empty."
            )

        if not self.query.strip():
            raise SharedContextValidationError(
                "SharedContext.query must not be blank."
            )

        return self
    # =========================================================================
    # OLD TOOL API COMPATIBILITY
    # =========================================================================

    def add_output(
        self,
        output,
        *,
        strict: bool = False,
    ) -> None:
        """
        Backward-compatible tool API.
        """
        with self._lock:

            if strict:
                exists = any(
                    o.step_id == output.step_id
                    for o in self.tool_outputs
                )

                if exists:
                    raise SharedContextValidationError(
                        f"Duplicate step_id: {output.step_id!r}"
                    )

            self.tool_outputs.append(output)

    def get_output(self, step_id: str):
        with self._lock:
            for output in self.tool_outputs:
                if output.step_id == step_id:
                    return output

        raise KeyError(f"No output for step_id={step_id!r}")

    def get_outputs_by_tool(self, tool_name: str):
        with self._lock:
            return [
                output
                for output in self.tool_outputs
                if output.tool_name == tool_name
            ]

    def all_outputs(self):
        with self._lock:
            return list(self.tool_outputs)

    # =========================================================================
    # Conversation helpers
    # =========================================================================

    def add_message(
        self,
        role: str,
        content: str,
        **metadata: Any,
    ) -> None:
        """
        Append a conversation message.
        """
        with self._lock:
            self.conversation_history.append(
                Message(
                    role=role,
                    content=content,
                    metadata=metadata,
                )
            )

    def get_messages(self) -> List[Message]:
        """
        Return safe copy of conversation history.
        """
        with self._lock:
            return list(self.conversation_history)

    # =========================================================================
    # Document helpers
    # =========================================================================

    def add_document(
        self,
        doc_id: str,
        source: str,
        content: str,
        **metadata: Any,
    ) -> None:
        """
        Add retrieved/supporting document.
        """
        with self._lock:
            self.documents.append(
                Document(
                    doc_id=doc_id,
                    source=source,
                    content=content,
                    metadata=metadata,
                )
            )

    def get_documents(self) -> List[Document]:
        """
        Return safe copy of documents.
        """
        with self._lock:
            return list(self.documents)

    # =========================================================================
    # Agent helpers
    # =========================================================================

    def store_agent_output(
        self,
        agent_name: str,
        result: Any,
        **metadata: Any,
    ) -> None:
        """
        Store raw agent output directly.

        DO NOT wrap outputs because downstream agents
        expect the original result types.
        """
        with self._lock:

            # Preserve metadata separately if provided
            if metadata:
                self.metadata.setdefault(
                    "_agent_output_metadata",
                    {}
                )[agent_name] = metadata

            # Store RAW result object
            self.agent_outputs[agent_name] = result

    def get_agent_output(
        self,
        agent_name: str,
    ) -> Optional[Any]:
        """
        Retrieve raw stored agent output.
        """
        with self._lock:
            return self.agent_outputs.get(agent_name)
    # =========================================================================
    # Summary helper
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        """
        Lightweight runtime snapshot for debugging/logging.
        """
        with self._lock:
            return {
                "run_id": self.run_id,
                "session_id": self.session_id,
                "agent_id": self.agent_id,
                "query": self.query[:120],
                "created_at": self.created_at.isoformat(),
                "messages": len(self.conversation_history),
                "documents": len(self.documents),
                "agent_outputs": list(self.agent_outputs.keys()),
                "tool_outputs": len(self.tool_outputs),
                "citations": len(self.citations),
                "policy_violations": len(self.policy_violations),
                "total_tokens": self.token_usage.total_tokens,
                "trace_steps": len(self.execution_trace),
            }



# ============================================================================
# Standalone Debug
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SharedContext — standalone debug")
    print("=" * 70)

    ctx = SharedContext(
        query="Explain multimodal transformers"
    )

    ctx.add_message(
        "user",
        "Explain multimodal transformers"
    )

    ctx.add_document(
        doc_id="doc_1",
        source="research_paper.pdf",
        content="ViLT combines image and text embeddings..."
    )

    ctx.store_agent_output(
        "retrieval_agent",
        {"documents_found": 5}
    )

    ctx.add_tool_output(
        step_id="step_1",
        tool_name="web_search",
        summary="Retrieved research papers",
        raw_data={"papers": 5},
    )

    ctx.add_citation(
        "https://arxiv.org"
    )

    ctx.token_usage.add(
        prompt=120,
        completion=45,
    )

    ctx.trace(
        source="retrieval_agent",
        step="document_retrieval",
        duration_ms=143.2,
    )

    print("\n[SUMMARY]")
    print(ctx.summary())

    print("\n✅ SharedContext debug complete.")
