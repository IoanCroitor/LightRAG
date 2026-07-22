"""Shared configuration for the ParseBench RAG evaluation suite.

All settings are overridable via environment variables or via the local
``.env`` file inside this directory (``edw_parsebench_eval/.env``).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

CORPUS_DIR = PROJECT_ROOT / "corpus"
GROUND_TRUTH_DIR = PROJECT_ROOT / "ground_truth"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Minimal .env loader (no third-party dependency).
# Loads edw_parsebench_eval/.env if present.
# Only fills variables that are not already present in os.environ.
# ---------------------------------------------------------------------------


def _load_local_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value


_load_local_env()

# ---------------------------------------------------------------------------
# LightRAG server (the system under test)
# ---------------------------------------------------------------------------

LIGHTRAG_BASE_URL = os.environ.get("LIGHTRAG_BASE_URL", "http://127.0.0.1:9621").rstrip("/")
LIGHTRAG_API_KEY = os.environ.get("LIGHTRAG_API_KEY", "")

# Query parameters used when evaluating
QUERY_MODE = os.environ.get("LIGHTRAG_QUERY_MODE", "mix")
QUERY_TOP_K = int(os.environ.get("LIGHTRAG_QUERY_TOP_K", "0")) or None
QUERY_CHUNK_TOP_K = int(os.environ.get("LIGHTRAG_QUERY_CHUNK_TOP_K", "0")) or None
QUERY_ENABLE_RERANK = (
    None if os.environ.get("LIGHTRAG_QUERY_ENABLE_RERANK", "") == ""
    else os.environ.get("LIGHTRAG_QUERY_ENABLE_RERANK", "").lower() == "true"
)
ENABLE_THINKING = os.environ.get("LLM_ENABLE_THINKING", "false").lower() == "true"
INCLUDE_CITATION_HIGHLIGHTS = os.environ.get("INCLUDE_CITATION_HIGHLIGHTS", "true").lower() == "true"

# ---------------------------------------------------------------------------
# LLM used for ground-truth generation + LLM-as-judge scoring.
# Configurable independently from the server via edw_parsebench_eval/.env
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "hy3"

LLM_BASE_URL = (
    os.environ.get("LLM_BASE_URL")
    or os.environ.get("OPENROUTER_BASE_URL")
    or OPENROUTER_BASE_URL
).rstrip("/")

LLM_API_KEY = (
    os.environ.get("LLM_API_KEY")
    or os.environ.get("OPENROUTER_API_KEY", "")
)

LLM_MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", LLM_MODEL)

# Ground-truth generation tuning
QUESTIONS_PER_DOC = int(os.environ.get("QUESTIONS_PER_DOC", "6"))
GT_MAX_CONTEXT_CHARS = int(os.environ.get("GT_MAX_CONTEXT_CHARS", "12000"))
GT_TEMPERATURE = float(os.environ.get("GT_TEMPERATURE", "0.3"))
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))

# Network / retry tuning
HTTP_TIMEOUT = float(os.environ.get("EVAL_HTTP_TIMEOUT", "120"))
UPLOAD_TIMEOUT = float(os.environ.get("EVAL_UPLOAD_TIMEOUT", "300"))
MAX_RETRIES = int(os.environ.get("EVAL_MAX_RETRIES", "3"))

# Parallel concurrency tuning
PARSEBENCH_WORKERS = int(
    os.environ.get("PARSEBENCH_WORKERS")
    or os.environ.get("EVAL_MAX_WORKERS")
    or "4"
)
GT_MAX_WORKERS = int(os.environ.get("GT_MAX_WORKERS") or PARSEBENCH_WORKERS)
INGEST_MAX_WORKERS = int(os.environ.get("INGEST_MAX_WORKERS") or PARSEBENCH_WORKERS)
EVAL_MAX_WORKERS = int(os.environ.get("EVAL_MAX_WORKERS") or PARSEBENCH_WORKERS)


def auth_headers() -> dict[str, str]:
    if LIGHTRAG_API_KEY:
        return {"X-API-Key": LIGHTRAG_API_KEY}
    return {}
