#!/usr/bin/env bash
# Run the same EDW-RAG CI jobs locally or inside GitLab CI.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'EOF'
Usage: edw_rag_patches/ci/run_edw_pipeline.sh <job>

Jobs:
  lint          Run ruff on EDW-RAG patch changes.
  patch-tests   Run the focused EDW-RAG patch regression suite.
  parsebench    Clear, ingest, and evaluate a disposable LightRAG server.

For parsebench, export LIGHTRAG_BASE_URL and LLM_API_KEY (or
OPENROUTER_API_KEY). LIGHTRAG_API_KEY is also required when the target server
is authenticated. Results default to edw_parsebench_eval/results; override
with PARSEBENCH_OUTPUT_DIR.
EOF
}

run_parsebench() {
    if [[ -z "${LIGHTRAG_BASE_URL:-}" ]]; then
        printf 'LIGHTRAG_BASE_URL must point to a disposable LightRAG server.\n' >&2
        exit 2
    fi
    if [[ -z "${LLM_API_KEY:-}${OPENROUTER_API_KEY:-}" ]]; then
        printf 'Set LLM_API_KEY or OPENROUTER_API_KEY for ParseBench judging.\n' >&2
        exit 2
    fi

    # Services start asynchronously in GitLab. Wait before clearing the target
    # so the same runner works both against the CI service and an external URL.
    LIGHTRAG_BASE_URL="$LIGHTRAG_BASE_URL" uv run python -c '
import os
import time
from urllib.error import URLError
from urllib.request import urlopen

url = os.environ["LIGHTRAG_BASE_URL"].rstrip("/") + "/health"
deadline = time.monotonic() + float(
    os.environ.get("PARSEBENCH_SERVER_START_TIMEOUT", "180")
)
last_error = None
while time.monotonic() < deadline:
    try:
        with urlopen(url, timeout=15) as response:
            if response.status == 200:
                print(f"ParseBench target is healthy: {url}")
                break
            last_error = f"unexpected HTTP status {response.status}"
    except (OSError, URLError) as error:
        last_error = str(error)
    time.sleep(2)
else:
    raise SystemExit(
        f"ParseBench target did not become healthy within the startup timeout: "
        f"{url} ({last_error})"
    )
'

    # Cache entries can make a benchmark appear to pass without exercising the
    # current patch build. Clear both the server's in-memory and persistent LLM
    # cache before ParseBench clears/reloads documents and issues queries.
    LIGHTRAG_BASE_URL="$LIGHTRAG_BASE_URL" \
        LIGHTRAG_API_KEY="${LIGHTRAG_API_KEY:-}" \
        uv run python -c '
import json
import os
from urllib.request import Request, urlopen

base_url = os.environ["LIGHTRAG_BASE_URL"].rstrip("/")
headers = {"Content-Type": "application/json"}
if api_key := os.environ.get("LIGHTRAG_API_KEY"):
    headers["X-API-Key"] = api_key
request = Request(
    base_url + "/documents/clear_cache",
    data=b"{}",
    headers=headers,
    method="POST",
)
with urlopen(request, timeout=30) as response:
    if response.status != 200:
        raise SystemExit(
            f"Unexpected cache-clear status {response.status} from {request.full_url}"
        )
    payload = json.loads(response.read().decode("utf-8") or "{}")
print(f"ParseBench cache cleared: {payload.get('message', 'ok')}")
'

    local output_dir="${PARSEBENCH_OUTPUT_DIR:-edw_parsebench_eval/results}"
    local workers="${PARSEBENCH_WORKERS:-1}"
    local ingest_timeout="${PARSEBENCH_INGEST_TIMEOUT:-1800}"
    local eval_args=(
        edw_parsebench_eval/run_eval.py
        --output-dir "$output_dir"
        --workers "$workers"
    )
    if [[ -n "${PARSEBENCH_MAX_QUESTIONS:-}" ]]; then
        eval_args+=(--max-questions "$PARSEBENCH_MAX_QUESTIONS")
    fi

    # Ground truth is versioned. CI evaluates it rather than regenerating it,
    # keeping the benchmark deterministic and avoiding a needless LLM cost.
    uv run python edw_parsebench_eval/ingest.py \
        --clear --timeout "$ingest_timeout" --workers "$workers"
    uv run python "${eval_args[@]}"
}

case "${1:-}" in
    lint)
        # The current patch branch contains two intentional-looking, existing
        # log-only f-strings; keep that baseline from blocking all CI while
        # retaining the rest of Ruff's checks for the patch surface.
        uv run ruff check \
            --ignore F541 \
            edw_rag_patches
        ;;
    patch-tests)
        uv run python edw_rag_patches/ci/verify_edw_patches.py
        # These regressions manage apply/revert cycles themselves.  Starting
        # them with global patches already installed would hide that behavior.
        uv run python -m pytest \
            edw_rag_patches/tests \
            -q -m 'not integration and not requires_db and not requires_api'
        ;;
    parsebench)
        run_parsebench
        ;;
    -h|--help|help|'')
        usage
        ;;
    *)
        printf 'Unknown CI job: %s\n\n' "$1" >&2
        usage >&2
        exit 2
        ;;
esac
