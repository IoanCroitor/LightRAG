"""Shared configuration for the ParseBench RAG evaluation suite.

All settings are overridable via environment variables so the suite can run
against any LightRAG (patched or upstream) deployment without code changes.

The script also loads the repository ``.env`` (the one next to this package's
parent directory) into ``os.environ`` the first time this module is imported,
so the GPUStack credentials used by the LightRAG server are reused here for
ground-truth generation and LLM-as-judge scoring.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

CORPUS_DIR = PROJECT_ROOT / "corpus"
GROUND_TRUTH_DIR = PROJECT_ROOT / "ground_truth"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Minimal .env loader (no third-party dependency).
# Only fills variables that are not already present in the environment.
# ---------------------------------------------------------------------------

def _load_repo_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_repo_env()


# ---------------------------------------------------------------------------
# LightRAG server (the system under test)
# ---------------------------------------------------------------------------

LIGHTRAG_BASE_URL = os.environ.get("LIGHTRAG_BASE_URL", "http://127.0.0.1:9621").rstrip("/")
# No AUTH_ACCOUNTS / LIGHTRAG_API_KEY is configured in this deployment, so the
# header is optional; set LIGHTRAG_API_KEY if your server enforces one.
LIGHTRAG_API_KEY = os.environ.get("LIGHTRAG_API_KEY", "")

# Query parameters used when evaluating (mirrors a realistic mix-mode query).
QUERY_MODE = os.environ.get("LIGHTRAG_QUERY_MODE", "mix")
QUERY_TOP_K = int(os.environ.get("LIGHTRAG_QUERY_TOP_K", "0")) or None
QUERY_CHUNK_TOP_K = int(os.environ.get("LIGHTRAG_QUERY_CHUNK_TOP_K", "0")) or None
QUERY_ENABLE_RERANK = (
    None if os.environ.get("LIGHTRAG_QUERY_ENABLE_RERANK", "") == ""
    else os.environ.get("LIGHTRAG_QUERY_ENABLE_RERANK", "").lower() == "true"
)
# Qwen3 emits a long chain-of-thought that consumes the token budget and can
# truncate the structured JSON we ask for. Disable thinking for deterministic
# JSON generation / judging (set LLM_ENABLE_THINKING=true to re-enable).
ENABLE_THINKING = os.environ.get("LLM_ENABLE_THINKING", "false").lower() == "true"
# The EDW-RAG patches add citation highlights to query responses; request them
# so we can also assert the patch is active in the deployment under test.
INCLUDE_CITATION_HIGHLIGHTS = os.environ.get("INCLUDE_CITATION_HIGHLIGHTS", "true").lower() == "true"

# ---------------------------------------------------------------------------
# LLM used for ground-truth generation + LLM-as-judge scoring.
# Reuses the GPUStack OpenAI-compatible endpoint the server already uses.
# ---------------------------------------------------------------------------

LLM_BASE_URL = os.environ.get("LLM_BINDING_HOST", "http://fdz2.edw.ro/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_BINDING_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b-fast")
# Judge may use a stronger/cheaper model; defaults to the same model.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", LLM_MODEL)

# Ground-truth generation tuning.
QUESTIONS_PER_DOC = int(os.environ.get("QUESTIONS_PER_DOC", "6"))
GT_MAX_CONTEXT_CHARS = int(os.environ.get("GT_MAX_CONTEXT_CHARS", "12000"))
GT_TEMPERATURE = float(os.environ.get("GT_TEMPERATURE", "0.3"))
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))

# Network / retry tuning.
HTTP_TIMEOUT = float(os.environ.get("EVAL_HTTP_TIMEOUT", "120"))
UPLOAD_TIMEOUT = float(os.environ.get("EVAL_UPLOAD_TIMEOUT", "300"))
MAX_RETRIES = int(os.environ.get("EVAL_MAX_RETRIES", "3"))


def auth_headers() -> dict[str, str]:
    """Return request headers that authenticate against the LightRAG API.

    Empty when no API key is configured (fully-open deployment).
    """
    if LIGHTRAG_API_KEY:
        return {"X-API-Key": LIGHTRAG_API_KEY}
    return {}
