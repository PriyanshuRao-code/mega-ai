# Multi-Agent System — Infrastructure

Production-grade, Dockerized multi-agent infrastructure.  
All configuration is **environment variable only** — no hard-coded secrets.

---

## Services & Ports

| Service    | Internal Port | Host Port (default) | Protocol | Purpose                        |
|------------|:-------------:|:-------------------:|----------|--------------------------------|
| `fluentd`  | 24224         | 24224               | TCP/UDP  | Structured log ingestion       |
| `postgres` | 5432          | 5432                | TCP      | Primary database               |
| `api`      | 8000          | 8000                | HTTP     | REST / health endpoint         |
| `worker`   | —             | —                   | —        | Internal task queue consumer   |

> Override any host port via `.env` (e.g. `API_HOST_PORT=9000`).

---

## Startup Order

```
fluentd  →  postgres  →  api  →  worker
   ↑            ↑           ↑        ↑
 (first)    (healthy    (healthy  (depends on
             pg_isready) /health)   api+pg)
```

1. **fluentd** — log sink starts first; all other services forward logs here.  
2. **postgres** — health-checked via `pg_isready`; `api` and `worker` won't start until healthy.  
3. **api** — waits for `postgres` (healthy) + `fluentd` (healthy); exposes `GET /health`.  
4. **worker** — waits for `api` (healthy) + `postgres` (healthy) + `fluentd` (healthy).

---

## Quick Start

```bash
# 1. Copy and fill in the env file
cp .env.example .env
$EDITOR .env          # set POSTGRES_PASSWORD, API_SECRET_KEY, etc.

# 2. Start everything
docker-compose up -d --build

# 3. Verify
docker-compose ps
docker-compose logs -f api

# 4. Run the debug validator
python debug/run_infra_debug.py --env-file .env --verbose
```

---

## Debug Script

```
debug/run_infra_debug.py
```

| Flag            | Purpose                                    |
|-----------------|--------------------------------------------|
| `--env-file`    | Path to .env (default: `.env`)             |
| `--verbose`     | Show detail for all checks, not just fails |
| `--json`        | Machine-readable JSON output               |
| `--skip-docker` | Skip Docker checks (useful in CI)          |

**Checks performed:**

1. Required environment variables (presence + no placeholder values)
2. Docker daemon reachability  
3. Per-container state + health (`postgres`, `api`, `worker`, `fluentd`)
4. PostgreSQL TCP + query (`SELECT version()`)
5. API `GET /health` HTTP 200
6. Fluentd port TCP reachability

**Exit codes:** `0` = all passed, `1` = one or more failed.

---

## Volumes

| Volume          | Mounted in       | Contents                      |
|-----------------|------------------|-------------------------------|
| `postgres_data` | `/var/lib/postgresql/data` | Durable DB files |
| `fluentd_logs`  | `/fluentd/log`   | Rotated, gzip-compressed JSON logs |

---

## Structured Logging

All services emit **JSON logs** captured by Fluentd via the Docker `fluentd` log driver.

Log files are persisted to the `fluentd_logs` volume, organised as:

```
/fluentd/log/<container_name>/YYYY-MM-DD.log.gz
```

Set `LOG_LEVEL` in `.env` to control verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

---

## Service Dependencies

```
api
├── postgres   (healthy)
└── fluentd    (healthy)

worker
├── api        (healthy)
├── postgres   (healthy)
└── fluentd    (healthy)

postgres
└── fluentd    (healthy)
```

---

## Environment Variables

See [`.env.example`](.env.example) for the full annotated list.

| Variable             | Required | Description                          |
|----------------------|:--------:|--------------------------------------|
| `POSTGRES_DB`        | ✔        | Database name                        |
| `POSTGRES_USER`      | ✔        | Database user                        |
| `POSTGRES_PASSWORD`  | ✔        | Database password                    |
| `API_SECRET_KEY`     | ✔        | API signing key (≥32 chars)          |
| `LOG_LEVEL`          | ✔        | Log level (INFO recommended)         |
| `API_WORKERS`        |          | Uvicorn workers (default: 2)         |
| `WORKER_CONCURRENCY` |          | Parallel worker tasks (default: 4)   |
| `API_HOST_PORT`      |          | Host port for API (default: 8000)    |
| `POSTGRES_HOST_PORT` |          | Host port for PG (default: 5432)     |
| `FLUENTD_HOST_PORT`  |          | Host port for Fluentd (default: 24224)|
