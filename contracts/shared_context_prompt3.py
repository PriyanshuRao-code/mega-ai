# contracts/shared_context.py (GENERATED ALONG WITH tools)

"""
contracts/shared_context.py
===========================
Mutable context object shared across all tools within one agent turn.

Imports   : dataclasses, typing, threading
Inputs    : (constructed by the agent orchestrator)
Outputs   : SharedContext
Exceptions: KeyError  — get_output() for unknown step_id
            ValueError — add_output() with duplicate step_id (strict mode)
Dependencies: stdlib only
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentOutput:
    """Record of a single previous tool invocation stored in context."""
    step_id   : str
    tool_name : str
    summary   : str
    raw_data  : Any
    metadata  : dict[str, Any] = field(default_factory=dict)


class SharedContext:
    """
    Thread-safe, append-only log of agent outputs within a single turn.

    Responsibilities
    ----------------
    - Store ordered history of tool outputs (used by SelfReflectionTool)
    - Provide typed retrieval by step_id or tool_name
    - Carry session-scoped key/value store for inter-tool communication

    Thread safety
    -------------
    All mutations guarded by an internal RLock so tools running in a
    thread pool cannot corrupt the context.
    """

    def __init__(self, session_id: str, agent_id: str) -> None:
        self.session_id  : str                       = session_id
        self.agent_id    : str                       = agent_id
        self._outputs    : list[AgentOutput]         = []
        self._kv_store   : dict[str, Any]            = {}
        self._lock       : threading.RLock           = threading.RLock()

    # ── output history ─────────────────────────────────────────────────── #

    def add_output(self, output: AgentOutput, *, strict: bool = False) -> None:
        """Append a tool output. Raises ValueError on duplicate step_id if strict=True."""
        with self._lock:
            if strict and any(o.step_id == output.step_id for o in self._outputs):
                raise ValueError(f"Duplicate step_id: {output.step_id!r}")
            self._outputs.append(output)

    def get_output(self, step_id: str) -> AgentOutput:
        """Return the output for *step_id*. Raises KeyError if not found."""
        with self._lock:
            for o in self._outputs:
                if o.step_id == step_id:
                    return o
        raise KeyError(f"No output for step_id={step_id!r}")

    def get_outputs_by_tool(self, tool_name: str) -> list[AgentOutput]:
        """Return all outputs produced by *tool_name*, in insertion order."""
        with self._lock:
            return [o for o in self._outputs if o.tool_name == tool_name]

    def all_outputs(self) -> list[AgentOutput]:
        """Return a snapshot of all stored outputs (safe copy)."""
        with self._lock:
            return list(self._outputs)

    # ── key-value store ───────────────────────────────────────────────── #

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._kv_store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._kv_store.get(key, default)

    # ── dunder helpers ────────────────────────────────────────────────── #

    def __len__(self) -> int:
        with self._lock:
            return len(self._outputs)

    def __repr__(self) -> str:
        return (
            f"SharedContext(session={self.session_id!r}, "
            f"agent={self.agent_id!r}, outputs={len(self)})"
        )
