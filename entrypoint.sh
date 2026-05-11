#!/usr/bin/env sh
# ============================================================
# entrypoint.sh
# Dispatches to the correct service based on $SERVICE env var.
# Called by the Dockerfile ENTRYPOINT.
# ============================================================

set -eu

SERVICE="${SERVICE:-api}"

log() {
    printf '{"time":"%s","level":"INFO","service":"%s","msg":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SERVICE" "$1"
}

log "Starting service: $SERVICE"

case "$SERVICE" in
    api)
        log "Launching API server on 0.0.0.0:${API_PORT:-8000}"
        exec uvicorn api.app:app \
            --host "${API_HOST:-0.0.0.0}" \
            --port "${API_PORT:-8000}" \
            --workers "${API_WORKERS:-2}" \
            --log-config logging.json \
            $([ "${API_RELOAD:-false}" = "true" ] && echo "--reload" || true)
        ;;
    worker)
        log "Launching worker with concurrency=${WORKER_CONCURRENCY:-4}"
        exec python -m orchestrator.orchestrator
        ;;
    *)
        printf 'ERROR: Unknown SERVICE=%s. Must be api|worker.\n' "$SERVICE" >&2
        exit 1
        ;;
esac
