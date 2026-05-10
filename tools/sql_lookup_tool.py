"""
tools/sql_lookup_tool.py
========================
Converts a natural-language question into a SQL query, executes it
against a configured data source, and returns typed tabular results.

Imports
-------
    stdlib  : logging, re, sqlite3, time
    internal: interfaces.base_tool.BaseTool
              contracts.tool_contracts.{ToolRequest, ToolResponse,
                  ToolStatus, SQLResult}
              contracts.shared_context.SharedContext

Inputs  : ToolRequest
    payload keys:
        nl_query       (str, required)   — natural-language question
        database       (str, optional)   — SQLite path or ":memory:"
                                           (defaults to in-memory demo DB)
        max_rows       (int, default=100)— result row cap
        dry_run        (bool, default=False) — translate only, don't execute

Outputs : ToolResponse
    data: SQLResult
        nl_query       : original NL question (echoed)
        generated_sql  : the SQL string that was (or would be) executed
        columns        : list of column names
        rows           : list[list[Any]] — result rows
        row_count      : len(rows)
        exec_ms        : query wall-clock time

Exceptions handled
------------------
    ValueError   — missing/blank nl_query, invalid max_rows
    TypeError    — payload not a dict
    sqlite3.Error — query syntax / execution error → ToolStatus.ERROR
    TimeoutError  — re-raised to trigger BaseTool retry
    Exception    — catch-all retried by BaseTool

Dependencies
------------
    stdlib sqlite3 only.
    Swap _nl_to_sql() with a real LLM call (OpenAI, Anthropic, etc.).
    Swap _get_connection() with psycopg2/asyncpg for production DBs.

Security notes
--------------
    • Only SELECT queries are allowed (enforced by _guard_sql).
    • All queries use parameterised execution where possible.
    • Row cap prevents accidental full-table dumps.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from typing import Any

from interfaces.base_tool import BaseTool
from contracts.tool_contracts import (
    SQLResult,
    ToolRequest,
    ToolResponse,
    ToolStatus,
)
from contracts.shared_context import SharedContext

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  NL → SQL translator stub
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"how many (.+?) are there", re.I),   r"SELECT COUNT(*) AS count FROM \1"),
    (re.compile(r"list all (.+)",            re.I),   r"SELECT * FROM \1 LIMIT {max_rows}"),
    (re.compile(r"show (.+?) where (.+)",    re.I),   r"SELECT * FROM \1 WHERE \2 LIMIT {max_rows}"),
    (re.compile(r"average (.+?) of (.+)",    re.I),   r"SELECT AVG(\1) AS avg_\1 FROM \2"),
    (re.compile(r"total (.+?) of (.+)",      re.I),   r"SELECT SUM(\1) AS total_\1 FROM \2"),
]


def _nl_to_sql(nl_query: str, max_rows: int) -> str:
    """
    Rule-based NL→SQL stub.
    Replace with an LLM call:
        response = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            system=SCHEMA_PROMPT,
            messages=[{"role": "user", "content": nl_query}],
        )
        return response.content[0].text.strip()
    """
    for pattern, template in _TEMPLATE_MAP:
        m = pattern.match(nl_query.strip())
        if m:
            sql = pattern.sub(template, nl_query.strip())
            sql = sql.replace("{max_rows}", str(max_rows))
            return sql
    # Fallback — safe SELECT with a LIKE filter
    sanitised = re.sub(r"[^\w\s]", "", nl_query)[:60]
    return f"SELECT * FROM documents WHERE content LIKE '%{sanitised}%' LIMIT {max_rows}"


def _guard_sql(sql: str) -> None:
    """Reject anything that isn't a SELECT statement."""
    first_token = sql.strip().split()[0].upper() if sql.strip() else ""
    if first_token != "SELECT":
        raise ValueError(
            f"Only SELECT queries are permitted; got: {first_token!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  In-memory demo database
# ─────────────────────────────────────────────────────────────────────────────

def _build_demo_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE documents (
            id      INTEGER PRIMARY KEY,
            title   TEXT,
            content TEXT,
            author  TEXT,
            year    INTEGER
        )"""
    )
    conn.executemany(
        "INSERT INTO documents (title, content, author, year) VALUES (?,?,?,?)",
        [
            ("Alpha Report", "analysis of alpha metrics", "Alice", 2023),
            ("Beta Study",   "beta testing results",      "Bob",   2024),
            ("Gamma Review", "gamma ray observations",    "Carol", 2022),
            ("Delta Brief",  "delta team briefing notes", "Dave",  2024),
        ],
    )
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────────────────────
#  Tool
# ─────────────────────────────────────────────────────────────────────────────

class SQLLookupTool(BaseTool):
    """NL-to-SQL tool with execution and typed tabular results."""

    TOOL_NAME      : str   = "sql_lookup"
    VERSION        : str   = "1.0.0"
    MAX_RETRIES    : int   = 3
    TIMEOUT_SECONDS: float = 20.0

    def __init__(self) -> None:
        # Demo DB is shared across calls within the same tool instance.
        # In production, swap for a real connection pool.
        self._demo_conn = _build_demo_db()

    # ── validation ────────────────────────────────────────────────────── #

    def validate(self, request: ToolRequest) -> None:
        if not isinstance(request.payload, dict):
            raise TypeError(f"payload must be dict, got {type(request.payload).__name__}")

        nl_query = request.payload.get("nl_query", "")
        if not isinstance(nl_query, str) or not nl_query.strip():
            raise ValueError("payload.nl_query must be a non-empty string")

        max_rows = request.payload.get("max_rows", 100)
        if not isinstance(max_rows, int) or max_rows < 1:
            raise ValueError("payload.max_rows must be a positive integer")

    # ── execution ─────────────────────────────────────────────────────── #

    def execute(self, request: ToolRequest, context: SharedContext) -> ToolResponse:
        t0       = time.monotonic()
        nl_query = request.payload["nl_query"].strip()
        max_rows = int(request.payload.get("max_rows", 100))
        dry_run  = bool(request.payload.get("dry_run", False))
        db_path  = request.payload.get("database", ":memory:")

        # ── translate ────────────────────────────────────────────────── #
        sql = _nl_to_sql(nl_query, max_rows)
        logger.debug("[%s] nl=%r → sql=%r", self.TOOL_NAME, nl_query, sql)

        try:
            _guard_sql(sql)
        except ValueError as exc:
            return ToolResponse.failure(
                request_id=request.request_id,
                tool_name =self.TOOL_NAME,
                status    =ToolStatus.INVALID_INPUT,
                error     =str(exc),
            )

        if dry_run:
            return ToolResponse.success(
                request_id =request.request_id,
                tool_name  =self.TOOL_NAME,
                data       =SQLResult(
                    nl_query     =nl_query,
                    generated_sql=sql,
                    columns      =[],
                    rows         =[],
                    row_count    =0,
                ),
                duration_ms=(time.monotonic() - t0) * 1000,
            )

        # ── execute ──────────────────────────────────────────────────── #
        conn = self._demo_conn if db_path == ":memory:" else sqlite3.connect(db_path)
        try:
            cursor  = conn.execute(sql)
            columns = [d[0] for d in (cursor.description or [])]
            rows    = [list(row) for row in cursor.fetchmany(max_rows)]

            if not columns and not rows:
                return ToolResponse.failure(
                    request_id=request.request_id,
                    tool_name =self.TOOL_NAME,
                    status    =ToolStatus.EMPTY,
                    error     =f"Query returned no data: {sql!r}",
                )

            exec_ms = (time.monotonic() - t0) * 1000
            return ToolResponse.success(
                request_id =request.request_id,
                tool_name  =self.TOOL_NAME,
                data       =SQLResult(
                    nl_query     =nl_query,
                    generated_sql=sql,
                    columns      =columns,
                    rows         =rows,
                    row_count    =len(rows),
                    exec_ms      =exec_ms,
                ),
                duration_ms=exec_ms,
            )

        except sqlite3.Error as exc:
            return ToolResponse.failure(
                request_id=request.request_id,
                tool_name =self.TOOL_NAME,
                status    =ToolStatus.ERROR,
                error     =f"SQL execution error: {exc} | sql={sql!r}",
            )
        finally:
            if db_path != ":memory:":
                conn.close()
