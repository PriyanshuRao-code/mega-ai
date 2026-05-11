# ============================================================
# Multi-Agent System — Base Image
# Targets  : api | worker
# Build arg: SERVICE=api|worker  (default: api)
#
# Port map  : api → 8000   worker → none (internal queue only)
# ============================================================

# ── Stage 1: builder ────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# C-extensions needed for psycopg2, uvloop, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ────────────────────────────────────────
FROM python:3.12-slim AS runtime

ARG SERVICE=api
ENV SERVICE=${SERVICE}

# Runtime-only libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Least-privilege non-root user
RUN groupadd -r appgroup \
 && useradd -r -g appgroup -d /app appuser

WORKDIR /app

# Installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Application source + config
COPY . .
RUN rm -rf .git .env docker-compose.yml Dockerfile  # Basic cleanup in case .dockerignore is missed
COPY logging.json ./logging.json
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
 && chown -R appuser:appgroup /app

USER appuser

# ── Health check ────────────────────────────────────────────
# api  → HTTP /health on :8000
# worker → trivial python exit-0 probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD sh -c ' \
        if [ "$SERVICE" = "api" ]; \
        then curl -sf http://localhost:8000/health; \
        else python -c "import sys; sys.exit(0)"; \
        fi'

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
