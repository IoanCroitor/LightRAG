"""EDW-RAG Gunicorn -- LightRAG with chunk-level citation highlights.

For production multi-worker deployments.  Applies EDW-RAG patches before
the Gunicorn workers are forked and wraps ``lightrag-gunicorn``.

Usage::

    uv run python -m edw_rag_patches.gunicorn --workers 4 --port 9621
"""

from __future__ import annotations


def main() -> None:
    """Entrypoint: apply patches, then delegate to lightrag-gunicorn."""
    import os
    import sys

    # ── 1. Apply EDW-RAG patches BEFORE Gunicorn workers fork ──────
    from edw_rag_patches import apply_edw_rag_patches

    apply_edw_rag_patches()

    # Signal to worker processes that patches are already applied
    os.environ["EDW_RAG_PATCHES_APPLIED"] = "1"

    # ── 2. Monkey-patch create_app to also patch routes ────────────
    import lightrag.api.lightrag_server as ls

    _orig_create_app = ls.create_app

    def _patched_create_app(args):
        app = _orig_create_app(args)
        from edw_rag_patches import patch_app_routes

        patch_app_routes(app)
        return app

    ls.create_app = _patched_create_app

    # ── 3. Patch get_application (used by Gunicorn's load()) ───────
    # get_application calls create_app internally; our create_app patch
    # is sufficient.

    # ── 4. Delegate to lightrag-gunicorn ───────────────────────────
    from lightrag.api.run_with_gunicorn import main as gunicorn_main

    sys.argv[0] = "edw-rag-gunicorn"
    gunicorn_main()


if __name__ == "__main__":
    main()
