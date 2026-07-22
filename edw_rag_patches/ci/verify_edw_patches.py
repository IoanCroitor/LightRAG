#!/usr/bin/env python3
"""Fail fast when EDW-RAG's monkey-patch targets drift upstream."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the source-tree overlay explicit when invoked through
# ``uv run python path/to/script.py``.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

# API router imports parse command-line options, so prevent CI command flags
# from being interpreted as server options while patches are installed.
sys.argv = [sys.argv[0]]

def main() -> int:
    from edw_rag_patches import apply_edw_rag_patches, revert_edw_rag_patches

    apply_edw_rag_patches()
    try:
        import lightrag.base as base
        import lightrag.operate as operate
        import lightrag.utils as utils
        import lightrag.utils_pipeline as utils_pipeline

        assert (
            utils_pipeline.build_chunks_dict_from_chunking_result.__module__
            == "edw_rag_patches"
        )
        assert operate._get_vector_context.__name__ == "get_vector_context"
        assert utils.convert_to_user_format.__module__ == "edw_rag_patches.query_chain"
        assert hasattr(base.QueryParam, "include_citation_highlights")
    finally:
        revert_edw_rag_patches()

    print("EDW-RAG patch targets verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
