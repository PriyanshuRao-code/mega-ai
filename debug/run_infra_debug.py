#!/usr/bin/env python3
"""
debug/run_infra_debug.py
════════════════════════════════════════════════════════════════
Infrastructure debug/validation script for the Multi-Agent System.

Checks performed
────────────────
1. Required environment variables (presence + non-empty)
2. Docker daemon reachability
3. Per-service container status (running, healthy)
4. PostgreSQL connectivity (TCP + query)
5. API /health endpoint reachability
6. Fluentd log port reachability

Usage
─────
    python debug/run_infra_debug.py [--env-file ../.env] [--verbose]

Exit codes
──────────
    0  — all checks passed
    1  — one or more checks failed
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

# ── Optional deps (import-guarded with friendly errors) ─────────────────────
try:
    import psycopg2  # type: ignore
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    import urllib.request as _urllib_request
    import urllib.error as _urllib_error
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    detail: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class DebugReport:
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    results: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"


def _tick(passed: bool) -> str:
    return f"{GREEN}✔{RESET}" if passed else f"{RED}✘{RESET}"


def _timed(fn: Callable[[], CheckResult]) -> CheckResult:
    t0 = time.perf_counter()
    result = fn()
    result.duration_ms = (time.perf_counter() - t0) * 1000
    return result


def load_env_file(path: str) -> None:
    """Minimal .env loader — sets vars only if not already exported."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Check 1: Required environment variables
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED_ENV_VARS: list[tuple[str, str]] = [
    # (VAR_NAME, human-readable purpose)
    ("POSTGRES_DB",       "PostgreSQL database name"),
    ("POSTGRES_USER",     "PostgreSQL username"),
    ("POSTGRES_PASSWORD", "PostgreSQL password"),
    ("API_SECRET_KEY",    "API signing secret (≥32 chars recommended)"),
    ("LOG_LEVEL",         "Log verbosity level"),
]

PLACEHOLDER_VALUES = {"CHANGE_ME", "changeme", "your_secret", "password", "secret"}


def check_env_vars() -> CheckResult:
    missing, weak = [], []
    for var, purpose in REQUIRED_ENV_VARS:
        val = os.environ.get(var, "")
        if not val:
            missing.append(f"{var} ({purpose})")
        elif any(p in val.upper() for p in {v.upper() for v in PLACEHOLDER_VALUES}):
            weak.append(var)

    if missing:
        return CheckResult(
            name="Environment variables",
            passed=False,
            message=f"{len(missing)} required variable(s) missing",
            detail="\n  Missing: " + "\n  Missing: ".join(missing),
        )
    if weak:
        return CheckResult(
            name="Environment variables",
            passed=False,
            message=f"{len(weak)} variable(s) still use placeholder values",
            detail="  Weak vars: " + ", ".join(weak),
        )
    return CheckResult(
        name="Environment variables",
        passed=True,
        message=f"All {len(REQUIRED_ENV_VARS)} required variables present",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Check 2: Docker daemon
# ═══════════════════════════════════════════════════════════════════════════════

def check_docker_daemon() -> CheckResult:
    rc, out, err = run_cmd(["docker", "info", "--format", "{{.ServerVersion}}"])
    if rc != 0:
        return CheckResult(
            name="Docker daemon",
            passed=False,
            message="Docker daemon unreachable",
            detail=err or "Is Docker Desktop / dockerd running?",
        )
    return CheckResult(
        name="Docker daemon",
        passed=True,
        message=f"Docker engine v{out}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Check 3: Container status
# ═══════════════════════════════════════════════════════════════════════════════

EXPECTED_SERVICES = [
    ("postgres", True),   # (service_name_fragment, has_healthcheck)
    ("api",      True),
    ("worker",   True),
    ("fluentd",  True),
]


def _container_info(fragment: str) -> Optional[dict]:
    """Return the first container whose name contains `fragment`."""
    rc, out, _ = run_cmd([
        "docker", "ps", "-a",
        "--format", "{{json .}}",
        "--filter", f"name={fragment}",
    ])
    if rc != 0 or not out:
        return None
    # docker ps can return multiple lines
    for line in out.splitlines():
        try:
            data = json.loads(line)
            if fragment.lower() in data.get("Names", "").lower():
                return data
        except json.JSONDecodeError:
            continue
    return None


def check_containers() -> list[CheckResult]:
    results = []
    for service, has_hc in EXPECTED_SERVICES:

        def _check(svc=service, hc=has_hc) -> CheckResult:
            info = _container_info(svc)
            if info is None:
                return CheckResult(
                    name=f"Container:{svc}",
                    passed=False,
                    message="Container not found — is docker-compose up?",
                )
            state  = info.get("State", "unknown")
            status = info.get("Status", "")
            running = state == "running"

            healthy = True
            hc_note = ""
            if hc and running:
                healthy = "healthy" in status.lower()
                hc_note = f" | health: {'healthy' if healthy else status}"

            passed = running and healthy
            return CheckResult(
                name=f"Container:{svc}",
                passed=passed,
                message=f"state={state}{hc_note}",
                detail=None if passed else f"Full status: {status}",
            )

        results.append(_timed(_check))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Check 4: PostgreSQL connectivity
# ═══════════════════════════════════════════════════════════════════════════════

def check_postgres() -> CheckResult:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_HOST_PORT", "5432"))
    db   = os.environ.get("POSTGRES_DB", "")
    user = os.environ.get("POSTGRES_USER", "")
    pw   = os.environ.get("POSTGRES_PASSWORD", "")

    # Step 1: TCP reachability
    try:
        with socket.create_connection((host, port), timeout=5):
            pass
    except OSError as e:
        return CheckResult(
            name="PostgreSQL:TCP",
            passed=False,
            message=f"Cannot reach {host}:{port}",
            detail=str(e),
        )

    # Step 2: psycopg2 query
    if not HAS_PSYCOPG2:
        return CheckResult(
            name="PostgreSQL:query",
            passed=True,
            message=f"TCP OK at {host}:{port} (psycopg2 not installed — skipping query check)",
        )

    dsn = f"host={host} port={port} dbname={db} user={user} password={pw} connect_timeout=5"
    try:
        conn = psycopg2.connect(dsn)
        cur  = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM pg_stat_activity WHERE state='active';")
        active = cur.fetchone()[0]
        cur.close()
        conn.close()
        return CheckResult(
            name="PostgreSQL:query",
            passed=True,
            message=f"Connected — {version.split(',')[0]}",
            detail=f"Active connections: {active}",
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name="PostgreSQL:query",
            passed=False,
            message="Query failed",
            detail=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Check 5: API /health endpoint
# ═══════════════════════════════════════════════════════════════════════════════

def check_api_health() -> CheckResult:
    host = os.environ.get("API_HOST", "localhost")
    port = int(os.environ.get("API_HOST_PORT", "8000"))
    url  = f"http://{host}:{port}/health"

    try:
        req  = _urllib_request.Request(url, headers={"Accept": "application/json"})
        resp = _urllib_request.urlopen(req, timeout=8)
        body = resp.read().decode()
        status = resp.getcode()
        return CheckResult(
            name="API:/health",
            passed=(status == 200),
            message=f"HTTP {status}",
            detail=body[:200] if body else None,
        )
    except _urllib_error.HTTPError as e:
        return CheckResult(
            name="API:/health",
            passed=False,
            message=f"HTTP {e.code}",
            detail=str(e.reason),
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name="API:/health",
            passed=False,
            message=f"Cannot reach {url}",
            detail=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Check 6: Fluentd port
# ═══════════════════════════════════════════════════════════════════════════════

def check_fluentd_port() -> CheckResult:
    host = os.environ.get("FLUENTD_HOST", "localhost")
    port = int(os.environ.get("FLUENTD_HOST_PORT", "24224"))
    try:
        with socket.create_connection((host, port), timeout=5):
            pass
        return CheckResult(
            name="Fluentd:TCP",
            passed=True,
            message=f"Port {port} reachable on {host}",
        )
    except OSError as e:
        return CheckResult(
            name="Fluentd:TCP",
            passed=False,
            message=f"Cannot reach {host}:{port}",
            detail=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(report: DebugReport, verbose: bool) -> None:
    width = 68
    print(f"\n{BOLD}{'═' * width}{RESET}")
    print(f"{BOLD}  MAS Infrastructure Debug Report{RESET}  {DIM}{report.timestamp}{RESET}")
    print(f"{BOLD}{'═' * width}{RESET}\n")

    for r in report.results:
        icon = _tick(r.passed)
        dur  = f"{DIM}({r.duration_ms:.0f}ms){RESET}" if r.duration_ms > 0 else ""
        print(f"  {icon}  {BOLD}{r.name:<28}{RESET}  {r.message}  {dur}")
        if r.detail and (verbose or not r.passed):
            for line in r.detail.splitlines():
                print(f"       {DIM}{line}{RESET}")

    print(f"\n{'─' * width}")
    total = len(report.results)
    passed = total - report.failed_count
    if report.passed:
        print(f"  {GREEN}{BOLD}ALL {total} CHECKS PASSED{RESET}")
    else:
        print(f"  {RED}{BOLD}{report.failed_count} / {total} CHECKS FAILED{RESET}")
    print(f"{'─' * width}\n")


def emit_json_report(report: DebugReport) -> None:
    payload = {
        "timestamp": report.timestamp,
        "passed": report.passed,
        "summary": {
            "total": len(report.results),
            "failed": report.failed_count,
        },
        "checks": [
            {
                "name": r.name,
                "passed": r.passed,
                "message": r.message,
                "detail": r.detail,
                "duration_ms": round(r.duration_ms, 2),
            }
            for r in report.results
        ],
    }
    print(json.dumps(payload, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate Multi-Agent System infrastructure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--env-file", default=".env",
                   help="Path to .env file (default: .env)")
    p.add_argument("--verbose", action="store_true",
                   help="Show detail for passing checks too")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON report instead of ANSI text")
    p.add_argument("--skip-docker", action="store_true",
                   help="Skip Docker daemon + container checks (useful in CI pipelines)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Load .env (or whatever path was passed)
    load_env_file(args.env_file)

    report = DebugReport()

    # ── 1. Env vars ─────────────────────────────────────────
    report.results.append(_timed(check_env_vars))

    # ── 2. Docker daemon + containers ───────────────────────
    if not args.skip_docker:
        report.results.append(_timed(check_docker_daemon))
        # Only probe containers if docker daemon is up
        if report.results[-1].passed:
            report.results.extend(check_containers())
    else:
        report.results.append(CheckResult(
            name="Docker checks",
            passed=True,
            message="Skipped via --skip-docker",
        ))

    # ── 3. PostgreSQL ────────────────────────────────────────
    report.results.append(_timed(check_postgres))

    # ── 4. API health ────────────────────────────────────────
    report.results.append(_timed(check_api_health))

    # ── 5. Fluentd port ──────────────────────────────────────
    report.results.append(_timed(check_fluentd_port))

    # ── Output ───────────────────────────────────────────────
    if args.json:
        emit_json_report(report)
    else:
        print_report(report, verbose=args.verbose)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
