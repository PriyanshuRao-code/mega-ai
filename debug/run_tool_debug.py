"""
debug/run_tool_debug.py
=======================
Standalone debug harness for the full tooling layer.

Imports
-------
    stdlib  : logging, sys, traceback, uuid, pathlib, json, os
    internal: all four tools, contracts, interfaces

Inputs  : (none — run as a script)
Outputs : coloured console report + exit code 0 (all pass) / 1 (any fail)

Exceptions: captures and reports all exceptions; never re-raises

Dependencies
------------
    All four tools must be importable (run from the project root):
        python -m debug.run_tool_debug
    or:
        cd project_root && python debug/run_tool_debug.py
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# ── make project root importable ──────────────────────────────────────────── #
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.shared_context import AgentOutput, SharedContext
from contracts.tool_contracts import (
    ExecutionResult,
    ReflectionResult,
    SQLResult,
    ToolRequest,
    ToolResponse,
    ToolStatus,
)
from tools.sandbox_tool import SandboxTool
from tools.self_reflection_tool import SelfReflectionTool
from tools.sql_lookup_tool import SQLLookupTool
from tools.web_search_tool import WebSearchTool

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("debug.harness")

# ─────────────────────────────────────────────────────────────────────────────
#  Console colours (ANSI — disabled on Windows without ENABLE_VIRTUAL_TERMINAL)
# ─────────────────────────────────────────────────────────────────────────────

_USE_COLOUR = sys.stdout.isatty() or os.environ.get("FORCE_COLOR")

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

GREEN  = lambda t: _c("32", t)   # noqa: E731
RED    = lambda t: _c("31", t)   # noqa: E731
YELLOW = lambda t: _c("33", t)   # noqa: E731
BOLD   = lambda t: _c("1",  t)   # noqa: E731
DIM    = lambda t: _c("2",  t)   # noqa: E731

# ─────────────────────────────────────────────────────────────────────────────
#  Test result record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name   : str
    passed : bool
    detail : str
    error  : str = ""

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _req(tool_name: str, payload: dict[str, Any], **kw: Any) -> ToolRequest:
    return ToolRequest(tool_name=tool_name, payload=payload, **kw)


def _ctx(seed: str = "debug") -> SharedContext:
    return SharedContext(session_id=f"sess-{seed}", agent_id=f"agent-{seed}")


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def _run_case(name: str, fn: Callable[[], None]) -> TestResult:
    try:
        fn()
        return TestResult(name=name, passed=True, detail="OK")
    except AssertionError as exc:
        return TestResult(name=name, passed=False, detail="ASSERTION FAILED", error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return TestResult(
            name=name,
            passed=False,
            detail="EXCEPTION",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )

# ─────────────────────────────────────────────────────────────────────────────
#  WebSearchTool cases
# ─────────────────────────────────────────────────────────────────────────────

def _web_happy_path() -> None:
    tool = WebSearchTool()
    req  = _req("web_search", {"query": "multi-agent systems production", "max_results": 3})
    resp = tool.run(req, _ctx())

    _assert(resp.ok,              f"Expected SUCCESS, got {resp.status}")
    _assert(resp.data is not None, "data must not be None")
    _assert(isinstance(resp.data, list), "data must be list[SearchResult]")
    _assert(len(resp.data) > 0,   "Must have at least one result")
    _assert(resp.data[0].score >= 0.0, "score must be >= 0")
    _assert(resp.data[0].rank >= 1,    "rank must be >= 1")
    _assert(resp.duration_ms >= 0,     "duration_ms must be non-negative")


def _web_malformed_input() -> None:
    tool = WebSearchTool()
    req  = _req("web_search", {"query": ""})   # blank query
    resp = tool.run(req, _ctx())

    _assert(resp.status == ToolStatus.INVALID_INPUT,
            f"Expected INVALID_INPUT, got {resp.status}")
    _assert(resp.error is not None, "error must be set on failure")
    _assert(resp.attempts == 1,     "Malformed input must not be retried")


def _web_min_score_filter() -> None:
    tool = WebSearchTool()
    # min_score=0.99 should filter out everything from the stub
    req  = _req("web_search", {"query": "xyz", "max_results": 3, "min_score": 0.99})
    resp = tool.run(req, _ctx())

    _assert(resp.status == ToolStatus.EMPTY,
            f"Expected EMPTY when all results filtered, got {resp.status}")


def _web_wrong_payload_type() -> None:
    tool = WebSearchTool()
    req  = ToolRequest(tool_name="web_search", payload="not a dict")  # type: ignore[arg-type]
    resp = tool.run(req, _ctx())

    _assert(resp.status == ToolStatus.INVALID_INPUT,
            f"Expected INVALID_INPUT for non-dict payload, got {resp.status}")


def _web_relevance_sorted() -> None:
    tool = WebSearchTool()
    req  = _req("web_search", {"query": "analysis metrics", "max_results": 5})
    resp = tool.run(req, _ctx())

    _assert(resp.ok, f"Expected SUCCESS, got {resp.status}")
    scores = [r.score for r in resp.data]
    _assert(scores == sorted(scores, reverse=True), "Results must be sorted by score desc")

# ─────────────────────────────────────────────────────────────────────────────
#  SandboxTool cases
# ─────────────────────────────────────────────────────────────────────────────

def _sandbox_happy_path() -> None:
    tool = SandboxTool()
    req  = _req("sandbox", {"code": "print('hello sandbox')"})
    resp = tool.run(req, _ctx())

    _assert(resp.ok, f"Expected SUCCESS, got {resp.status}")
    result: ExecutionResult = resp.data
    _assert("hello sandbox" in result.stdout, "stdout must contain print output")
    _assert(result.return_code == 0,          "return_code must be 0")
    _assert(not result.timed_out,             "timed_out must be False")


def _sandbox_stderr_capture() -> None:
    tool = SandboxTool()
    code = "import sys; sys.stderr.write('err output\\n')"
    req  = _req("sandbox", {"code": code})
    resp = tool.run(req, _ctx())

    _assert(resp.ok, f"Expected SUCCESS, got {resp.status}")
    _assert("err output" in resp.data.stderr, "stderr must be captured")


def _sandbox_nonzero_exit() -> None:
    tool = SandboxTool()
    req  = _req("sandbox", {"code": "raise SystemExit(42)"})
    resp = tool.run(req, _ctx())

    _assert(resp.ok, f"Expected SUCCESS (process ran), got {resp.status}")
    _assert(resp.data.return_code == 42, f"Expected rc=42, got {resp.data.return_code}")


def _sandbox_timeout() -> None:
    tool = SandboxTool()
    req  = _req("sandbox", {"code": "import time; time.sleep(60)", "timeout": 1.0})
    resp = tool.run(req, _ctx())

    _assert(resp.status == ToolStatus.TIMEOUT,
            f"Expected TIMEOUT, got {resp.status}")
    _assert(resp.data.timed_out,  "ExecutionResult.timed_out must be True")


def _sandbox_malformed_input() -> None:
    tool = SandboxTool()
    req  = _req("sandbox", {"code": "   "})   # whitespace-only
    resp = tool.run(req, _ctx())

    _assert(resp.status == ToolStatus.INVALID_INPUT,
            f"Expected INVALID_INPUT, got {resp.status}")


def _sandbox_multi_line_code() -> None:
    tool = SandboxTool()
    code = """
x = [i**2 for i in range(5)]
print(sum(x))
"""
    req  = _req("sandbox", {"code": code})
    resp = tool.run(req, _ctx())

    _assert(resp.ok, f"Expected SUCCESS, got {resp.status}")
    _assert("30" in resp.data.stdout, f"Expected '30' in stdout, got {resp.data.stdout!r}")

# ─────────────────────────────────────────────────────────────────────────────
#  SQLLookupTool cases
# ─────────────────────────────────────────────────────────────────────────────

def _sql_happy_path() -> None:
    tool = SQLLookupTool()
    req  = _req("sql_lookup", {"nl_query": "list all documents"})
    resp = tool.run(req, _ctx())

    _assert(resp.ok, f"Expected SUCCESS, got {resp.status}")
    result: SQLResult = resp.data
    _assert(result.nl_query == "list all documents", "nl_query must be echoed")
    _assert(len(result.generated_sql) > 0,           "generated_sql must not be empty")
    _assert(isinstance(result.columns, list),         "columns must be a list")
    _assert(result.row_count >= 0,                    "row_count must be non-negative")


def _sql_dry_run() -> None:
    tool = SQLLookupTool()
    req  = _req("sql_lookup", {"nl_query": "how many documents are there", "dry_run": True})
    resp = tool.run(req, _ctx())

    _assert(resp.ok, f"Expected SUCCESS (dry_run), got {resp.status}")
    _assert(resp.data.rows == [],    "dry_run must not return rows")
    _assert(len(resp.data.generated_sql) > 0, "generated_sql must be populated even in dry_run")


def _sql_malformed_input() -> None:
    tool = SQLLookupTool()
    req  = _req("sql_lookup", {"nl_query": ""})
    resp = tool.run(req, _ctx())

    _assert(resp.status == ToolStatus.INVALID_INPUT,
            f"Expected INVALID_INPUT, got {resp.status}")


def _sql_count_query() -> None:
    tool = SQLLookupTool()
    req  = _req("sql_lookup", {"nl_query": "how many documents are there"})
    resp = tool.run(req, _ctx())

    _assert(resp.ok, f"Expected SUCCESS, got {resp.status}")
    _assert(resp.data.row_count == 1, "COUNT query should return exactly 1 row")


def _sql_result_fields() -> None:
    tool = SQLLookupTool()
    req  = _req("sql_lookup", {"nl_query": "list all documents"})
    resp = tool.run(req, _ctx())

    _assert(resp.ok, f"Expected SUCCESS, got {resp.status}")
    _assert("title" in resp.data.columns or len(resp.data.columns) > 0,
            "columns must be populated")
    _assert(resp.data.exec_ms >= 0, "exec_ms must be non-negative")

# ─────────────────────────────────────────────────────────────────────────────
#  SelfReflectionTool cases
# ─────────────────────────────────────────────────────────────────────────────

def _make_context_with_outputs() -> SharedContext:
    ctx = _ctx("reflect")
    ctx.add_output(AgentOutput(
        step_id  ="step-1",
        tool_name="web_search",
        summary  ="Search succeeded. Found 5 valid results about machine learning.",
        raw_data =None,
    ))
    ctx.add_output(AgentOutput(
        step_id  ="step-2",
        tool_name="sql_lookup",
        summary  ="Query succeeded. Found 12 matching records in database.",
        raw_data =None,
    ))
    return ctx


def _make_contradictory_context() -> SharedContext:
    ctx = _ctx("contradict")
    ctx.add_output(AgentOutput(
        step_id  ="step-A",
        tool_name="web_search",
        summary  ="Search succeeded. Retrieved confirmed valid results for neural network analysis.",
        raw_data =None,
    ))
    ctx.add_output(AgentOutput(
        step_id  ="step-B",
        tool_name="sql_lookup",
        summary  ="Search failed. No confirmed results found. Neural network analysis is invalid and incorrect.",
        raw_data =None,
    ))
    return ctx


def _reflect_no_contradictions() -> None:
    tool = SelfReflectionTool()
    ctx  = _make_context_with_outputs()
    req  = _req("self_reflection", {"sensitivity": 0.1})
    resp = tool.run(req, _ctx())  # uses fresh empty context intentionally for empty test
    # override: use real context
    resp = tool.run(req, ctx)

    _assert(resp.ok, f"Expected SUCCESS, got {resp.status}")
    result: ReflectionResult = resp.data
    _assert(0.0 <= result.confidence_score <= 1.0, "confidence_score must be in [0,1]")
    _assert(isinstance(result.summary, str) and len(result.summary) > 0,
            "summary must be a non-empty string")


def _reflect_contradiction_detected() -> None:
    tool = SelfReflectionTool()
    ctx  = _make_contradictory_context()
    # sensitivity=0.05 ensures the shared-keyword overlap ratio clears the threshold
    req  = _req("self_reflection", {"sensitivity": 0.05})
    resp = tool.run(req, ctx)

    _assert(resp.ok, f"Expected SUCCESS, got {resp.status}")
    result: ReflectionResult = resp.data
    _assert(len(result.contradictions) > 0,
            "Contradictory outputs must yield at least one contradiction")
    _assert(result.confidence_score < 1.0, "Confidence must be < 1.0 when contradictions exist")


def _reflect_empty_context() -> None:
    tool = SelfReflectionTool()
    req  = _req("self_reflection", {})
    resp = tool.run(req, _ctx("empty"))

    _assert(resp.status == ToolStatus.EMPTY,
            f"Expected EMPTY for empty context, got {resp.status}")


def _reflect_step_ids_filter() -> None:
    tool = SelfReflectionTool()
    ctx  = _make_context_with_outputs()
    req  = _req("self_reflection", {"step_ids": ["step-1"], "sensitivity": 0.1})
    resp = tool.run(req, ctx)

    _assert(resp.ok, f"Expected SUCCESS, got {resp.status}")


def _reflect_invalid_step_id() -> None:
    tool = SelfReflectionTool()
    ctx  = _make_context_with_outputs()
    req  = _req("self_reflection", {"step_ids": ["nonexistent-step"]})
    resp = tool.run(req, ctx)

    _assert(resp.status == ToolStatus.INVALID_INPUT,
            f"Expected INVALID_INPUT for bad step_id, got {resp.status}")


def _reflect_malformed_sensitivity() -> None:
    tool = SelfReflectionTool()
    ctx  = _make_context_with_outputs()
    req  = _req("self_reflection", {"sensitivity": 99.9})   # out of range
    resp = tool.run(req, ctx)

    _assert(resp.status == ToolStatus.INVALID_INPUT,
            f"Expected INVALID_INPUT for bad sensitivity, got {resp.status}")

# ─────────────────────────────────────────────────────────────────────────────
#  Retry validation
# ─────────────────────────────────────────────────────────────────────────────

def _retry_counts_tracked() -> None:
    """
    Verify that BaseTool.run() reports the correct attempt count.
    Malformed input must NOT be retried (attempts=1).
    """
    tool = WebSearchTool()
    req  = _req("web_search", {"query": 12345})   # wrong type — invalid input
    resp = tool.run(req, _ctx())

    _assert(resp.status == ToolStatus.INVALID_INPUT, "Must be INVALID_INPUT")
    _assert(resp.attempts == 1, f"Malformed input must not be retried; got attempts={resp.attempts}")


def _retry_successful_response_has_attempts() -> None:
    tool = WebSearchTool()
    req  = _req("web_search", {"query": "retry validation test"})
    resp = tool.run(req, _ctx())

    _assert(resp.ok, f"Expected SUCCESS, got {resp.status}")
    _assert(resp.attempts >= 1, f"attempts must be >= 1, got {resp.attempts}")

# ─────────────────────────────────────────────────────────────────────────────
#  ToolResponse contract invariants
# ─────────────────────────────────────────────────────────────────────────────

def _response_contract_success() -> None:
    tool = SandboxTool()
    req  = _req("sandbox", {"code": "print(42)"})
    resp = tool.run(req, _ctx())

    _assert(resp.ok,                          ".ok must be True on SUCCESS")
    _assert(resp.request_id == req.request_id, "request_id must be echoed")
    _assert(resp.tool_name  == "sandbox",      "tool_name must be set")
    _assert(resp.timestamp  is not None,       "timestamp must be set")


def _response_contract_failure() -> None:
    resp = ToolResponse.failure(
        request_id="test-123",
        tool_name ="test",
        error     ="something went wrong",
    )
    _assert(not resp.ok,             ".ok must be False on failure")
    _assert(resp.data is None,       "data must be None on failure")
    _assert(resp.error is not None,  "error must be set")

# ─────────────────────────────────────────────────────────────────────────────
#  Test registry
# ─────────────────────────────────────────────────────────────────────────────

ALL_CASES: list[tuple[str, Callable[[], None]]] = [
    # WebSearchTool
    ("web_search / happy path",             _web_happy_path),
    ("web_search / malformed input",        _web_malformed_input),
    ("web_search / min_score filter",       _web_min_score_filter),
    ("web_search / wrong payload type",     _web_wrong_payload_type),
    ("web_search / relevance sorted",       _web_relevance_sorted),
    # SandboxTool
    ("sandbox    / happy path",             _sandbox_happy_path),
    ("sandbox    / stderr capture",         _sandbox_stderr_capture),
    ("sandbox    / nonzero exit code",      _sandbox_nonzero_exit),
    ("sandbox    / timeout",                _sandbox_timeout),
    ("sandbox    / malformed input",        _sandbox_malformed_input),
    ("sandbox    / multi-line code",        _sandbox_multi_line_code),
    # SQLLookupTool
    ("sql_lookup / happy path",             _sql_happy_path),
    ("sql_lookup / dry run",                _sql_dry_run),
    ("sql_lookup / malformed input",        _sql_malformed_input),
    ("sql_lookup / count query",            _sql_count_query),
    ("sql_lookup / result fields",          _sql_result_fields),
    # SelfReflectionTool
    ("reflection / no contradictions",     _reflect_no_contradictions),
    ("reflection / contradiction detected", _reflect_contradiction_detected),
    ("reflection / empty context",          _reflect_empty_context),
    ("reflection / step_ids filter",        _reflect_step_ids_filter),
    ("reflection / invalid step_id",        _reflect_invalid_step_id),
    ("reflection / malformed sensitivity",  _reflect_malformed_sensitivity),
    # Retry behaviour
    ("retry      / malformed not retried",  _retry_counts_tracked),
    ("retry      / success has attempts",   _retry_successful_response_has_attempts),
    # Contract invariants
    ("contract   / success invariants",     _response_contract_success),
    ("contract   / failure invariants",     _response_contract_failure),
]

# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all() -> int:
    """Execute all test cases and print a summary. Returns exit code."""
    print(BOLD("\n══════════════════════════════════════════"))
    print(BOLD("  Multi-Agent Tooling Layer — Debug Suite"))
    print(BOLD("══════════════════════════════════════════\n"))

    results: list[TestResult] = []
    for name, fn in ALL_CASES:
        res = _run_case(name, fn)
        results.append(res)
        status = GREEN("✔ PASS") if res.passed else RED("✘ FAIL")
        print(f"  {status}  {name}")
        if not res.passed:
            for line in res.error.splitlines():
                print(DIM(f"           {line}"))

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print(BOLD(f"\n══ Results: {passed}/{len(results)} passed  "
               f"{'· ' + RED(f'{failed} failed') if failed else GREEN('· all passed')} ══\n"))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
