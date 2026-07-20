"""
EDW-RAG Server -- LightRAG with chunk-level citation highlights.

CLI entrypoint that wraps ``lightrag-server`` and applies the
EDW-RAG patches before the app starts.

Usage::

    # Production (single process):
    uv run python -m edw_rag_patches.cli --host 0.0.0.0 --port 9621

    # Development (reload):
    uv run python -m edw_rag_patches.cli --reload --port 9621

    # With Gunicorn (multi-worker):
    uv run python -m edw_rag_patches.gunicorn --workers 4 --port 9621

All ``lightrag-server`` CLI flags are supported -- see ``--help``.
"""

from __future__ import annotations

import sys


def _load_dotenv_override() -> None:
    """Load ``.env`` with ``override=True`` before anything else.

    LightRAG's provider modules call ``load_dotenv(override=False)``,
    which refuses to replace an existing (even empty) env var.  We
    force-override here so ``.env`` values always take effect.
    """
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env", override=True)


def main() -> None:
    """Entrypoint: apply patches, delegate to lightrag-server."""

    # -- 0. Force-load .env before any LightRAG import reads env vars --
    _load_dotenv_override()

    # -- 1. Apply EDW-RAG patches BEFORE importing the server ---------
    from edw_rag_patches import apply_edw_rag_patches

    apply_edw_rag_patches()
    print("[edw-rag] Patches applied", flush=True)

    # -- 2. Strip EDW-RAG-specific flags from argv --------------------
    filtered_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--patches-json" and i + 1 < len(sys.argv):
            i += 2
            continue
        filtered_argv.append(arg)
        i += 1
    sys.argv = filtered_argv

    # -- 3. Monkey-patch server internals BEFORE they get imported ----
    import lightrag.api.lightrag_server as ls
    import lightrag.llm.openai as llo

    # 3a. Patch openai_complete_if_cache to inject minimum max_tokens.
    #     The Qwen reasoning model spends 4K+ tokens on thinking before
    #     producing content; the default max_tokens (if any) is too low.
    _orig_complete = llo.openai_complete_if_cache

    async def _patched_complete(
        model, prompt, system_prompt=None, history_messages=None,
        base_url=None, api_key=None, **kwargs,
    ):
        mt = kwargs.get("max_tokens")
        if mt is None or (isinstance(mt, int) and mt < 8192):
            kwargs["max_tokens"] = 8192
        return await _orig_complete(
            model, prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=base_url, api_key=api_key, **kwargs,
        )

    llo.openai_complete_if_cache = _patched_complete

    # 3b. Patch create_app to also call patch_app_routes() after the
    #     FastAPI app is assembled.
    _orig_create_app = ls.create_app

    def _patched_create_app(args):
        import sys as _sys
        _sys.stderr.write("[edw-rag-debug] _patched_create_app CALLED\n")
        _sys.stderr.flush()
        app = _orig_create_app(args)
        from edw_rag_patches import patch_app_routes
        _sys.stderr.write("[edw-rag-debug] patch_app_routes about to run\n")
        _sys.stderr.flush()
        patch_app_routes(app)
        _sys.stderr.write("[edw-rag-debug] patch_app_routes done\n")
        _sys.stderr.flush()
        return app

    ls.create_app = _patched_create_app

    # -- 4. Run the normal server startup -----------------------------
    ls.main()


if __name__ == "__main__":
    main()
