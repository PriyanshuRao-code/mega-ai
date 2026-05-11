# contracts/shared_context.py (GENERATED ALONG WITH agents)

"""
contracts/shared_context.py
============================
SharedContext — the single mutable context object threaded through the
entire multi-agent pipeline.

SOLID Alignment:
  - (S) Single Responsibility: holds pipeline state only, no logic
  - (O) Open/Closed: extend via optional fields; never remove existing ones

Imports:
  - dataclasses  (stdlib)
  - typing       (stdlib)
  - uuid         (stdlib)
  - datetime     (stdlib)

Exceptions:
  - SharedContextValidationError: raised when required fields are missing
                                  or have incompatible types
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class SharedContextValidationError(ValueError):
    """Raised when SharedContext fields fail validation."""


# ---------------------------------------------------------------------------
# Nested supporting types
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """
    A single conversational turn.

    Fields:
        role    : 'user' | 'assistant' | 'system'
        content : raw text of the message
        metadata: arbitrary extra data (timestamps, token counts, etc.)
    """
    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant", "system"}:
            raise SharedContextValidationError(
                f"Message.role must be 'user', 'assistant', or 'system'; got '{self.role}'"
            )
        if not isinstance(self.content, str):
            raise SharedContextValidationError("Message.content must be a string")


@dataclass
class Document:
    """
    A retrieved or injected document.

    Fields:
        doc_id   : unique document identifier
        source   : origin URI / path / label
        content  : full text body
        metadata : provenance, score, chunk info, etc.
    """
    doc_id: str
    source: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SharedContext
# ---------------------------------------------------------------------------

@dataclass
class SharedContext:
    """
    Pipeline-wide context object passed to every agent.

    Required fields:
        query           : the top-level user query driving this pipeline run

    Auto-populated fields (set at construction):
        run_id          : globally unique run identifier (UUID4)
        created_at      : UTC timestamp of context creation

    Optional fields (populated progressively by agents):
        conversation_history : ordered list of Message objects
        documents            : documents available for retrieval/synthesis
        agent_outputs        : keyed results from completed agents
        metadata             : arbitrary pipeline-level metadata

    Raises:
        SharedContextValidationError: if query is empty or types are invalid
    """

    # Required
    query: str

    # Auto-populated
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Progressive / optional
    conversation_history: List[Message] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise SharedContextValidationError("SharedContext.query must be a non-empty string")
        if not isinstance(self.conversation_history, list):
            raise SharedContextValidationError("conversation_history must be a list")
        if not isinstance(self.documents, list):
            raise SharedContextValidationError("documents must be a list")
        if not isinstance(self.agent_outputs, dict):
            raise SharedContextValidationError("agent_outputs must be a dict")

    # ------------------------------------------------------------------ #
    #  Convenience helpers (read-only; agents must not bypass contracts)  #
    # ------------------------------------------------------------------ #

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Append a message to conversation_history."""
        self.conversation_history.append(Message(role=role, content=content, metadata=metadata))

    def add_document(self, doc_id: str, source: str, content: str, **metadata: Any) -> None:
        """Append a document to the documents list."""
        self.documents.append(Document(doc_id=doc_id, source=source, content=content, metadata=metadata))

    def store_agent_output(self, agent_name: str, result: Any) -> None:
        """Persist an agent result into agent_outputs keyed by agent class name."""
        self.agent_outputs[agent_name] = result

    def get_agent_output(self, agent_name: str) -> Optional[Any]:
        """Retrieve a previously stored agent result, or None."""
        return self.agent_outputs.get(agent_name)

    def summary(self) -> Dict[str, Any]:
        """Return a lightweight dict snapshot for logging / debugging."""
        return {
            "run_id": self.run_id,
            "query": self.query[:120],
            "created_at": self.created_at.isoformat(),
            "messages": len(self.conversation_history),
            "documents": len(self.documents),
            "agent_outputs_keys": list(self.agent_outputs.keys()),
        }
