"""
EDW-RAG Patches -- chunk-level citation overlay for LightRAG

See ``edw_rag_patches/README.md`` for full documentation.
"""

from __future__ import annotations

import contextvars
import dataclasses
from typing import Any

_originals: dict[str, Any] = {}
_orig_build_chunks_dict = None

_citation_cv: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "_edw_citation_highlights", default=None
)
_blocks_path_cv: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_edw_blocks_path", default=None
)


def _save(name: str, mod: Any, attr: str) -> None:
    _originals[name] = getattr(mod, attr, None)


# ===================================================================
# Public API
# ===================================================================


def apply_edw_rag_patches() -> None:
    """Install all EDW-RAG patches.  Call once at startup."""
    global _orig_build_chunks_dict

    import lightrag.utils_pipeline as lup
    import lightrag.operate as lo
    import lightrag.utils as lu
    import lightrag.lightrag as ll
    import lightrag.base as lb
    import lightrag.pipeline as lpipe
    import lightrag.api.routers.query_routes as qr

    # -- 1. Storage: enrich chunks with positions at insert time -----------
    _save("build_chunks_dict_from_chunking_result", lup,
          "build_chunks_dict_from_chunking_result")
    _orig_build_chunks_dict = lup.build_chunks_dict_from_chunking_result
    lup.build_chunks_dict_from_chunking_result = _build_chunks_dict_patched

    # -- 1b. Vector Storage: preserve positions field in meta_fields ---------
    _save("lightrag_init", ll.LightRAG, "__init__")
    _orig_rag_init = ll.LightRAG.__init__

    def _rag_init_wrapper(self, *args, **kwargs):
        _orig_rag_init(self, *args, **kwargs)
        if hasattr(self, "chunks_vdb") and hasattr(self.chunks_vdb, "meta_fields"):
            if isinstance(self.chunks_vdb.meta_fields, set):
                self.chunks_vdb.meta_fields.add("positions")
            elif isinstance(self.chunks_vdb.meta_fields, (list, tuple)):
                self.chunks_vdb.meta_fields = set(self.chunks_vdb.meta_fields) | {"positions"}

    ll.LightRAG.__init__ = _rag_init_wrapper
    # -- 2. Query pipeline: propagate positions through retrieval ----------
    from . import query_chain as qc

    _save("_get_vector_context", lo, "_get_vector_context")
    lo._get_vector_context = qc.get_vector_context

    _save("_merge_all_chunks", lo, "_merge_all_chunks")
    lo._merge_all_chunks = qc.merge_all_chunks

    # -- 3. Response assembly: build citation_highlights sidecar -----------
    _save("convert_to_user_format", lu, "convert_to_user_format")
    lu.convert_to_user_format = qc.convert_to_user_format
    _save("operate_convert_to_user_format", lo, "convert_to_user_format")
    lo.convert_to_user_format = qc.convert_to_user_format
    _save("lightrag_convert_to_user_format", ll, "convert_to_user_format")
    ll.convert_to_user_format = qc.convert_to_user_format

    # -- 4. Pipeline: bridge blocks_path to chunk assembly -----------------
    _patch_pipeline(lpipe)

    # -- 5. API models -----------------------------------------------------
    _patch_query_param(lb)
    _patch_request_model(qr)
    _patch_response_model(qr)

    # -- 5b. Fix cross-module references -----------------------------------
    # ``query_routes.py`` does ``from lightrag.base import QueryParam`` at
    # module level.  That loads the ORIGINAL class before our patch above.
    # We must overwrite the local reference so ``to_query_params`` uses it.
    qr.QueryParam = lb.QueryParam

    # -- 6. Patch aquery_llm to bubble citation_highlights up --------------
    _save("aquery_llm", ll.LightRAG, "aquery_llm")
    _orig_aql = ll.LightRAG.aquery_llm

    async def _aql_wrapper(self, query: str, param=None, **kwargs):
        result = await _orig_aql(self, query, param=param, **kwargs)
        data = result.get("data", {})
        if isinstance(data, dict) and "citation_highlights" in data:
            result["citation_highlights"] = data["citation_highlights"]
            _citation_cv.set(data["citation_highlights"])
        return result

    ll.LightRAG.aquery_llm = _aql_wrapper


def revert_edw_rag_patches() -> None:
    """Restore all original LightRAG code."""
    import lightrag.utils_pipeline as lup
    import lightrag.operate as lo
    import lightrag.utils as lu
    import lightrag.lightrag as ll
    import lightrag.base as lb
    import lightrag.pipeline as lpipe
    import lightrag.api.routers.query_routes as qr

    restore = {
        "build_chunks_dict_from_chunking_result": (lup, "build_chunks_dict_from_chunking_result"),
        "pipeline_build_chunks_dict_from_chunking_result": (
            lpipe,
            "build_chunks_dict_from_chunking_result",
        ),
        "process_single_document": (lpipe._PipelineMixin, "process_single_document"),
        "_get_vector_context": (lo, "_get_vector_context"),
        "_merge_all_chunks": (lo, "_merge_all_chunks"),
        "convert_to_user_format": (lu, "convert_to_user_format"),
        "operate_convert_to_user_format": (lo, "convert_to_user_format"),
        "lightrag_convert_to_user_format": (ll, "convert_to_user_format"),
        "lightrag_init": (ll.LightRAG, "__init__"),
        "aquery_llm": (ll.LightRAG, "aquery_llm"),
    }
    for name, (mod, attr) in restore.items():
        if name in _originals and _originals[name] is not None:
            setattr(mod, attr, _originals.pop(name))
    for mod_name, orig_name in [(lb, "QueryParam"), (qr, "QueryRequest"),
                                (qr, "QueryResponse")]:
        if orig_name in _originals:
            setattr(mod_name, orig_name, _originals.pop(orig_name))


def patch_app_routes(app: Any) -> None:
    """Patch FastAPI app routes to include citation_highlights in /query."""
    import lightrag.api.routers.query_routes as qr

    for route in app.routes:
        if not hasattr(route, "methods") or not hasattr(route, "endpoint"):
            continue
        path = getattr(route, "path", "")
        matched = path.rstrip("/").endswith(("/query", "/query/stream"))
        if matched and "POST" in route.methods:
            if hasattr(route, "response_model") and route.response_model is not None:
                route.response_model = qr.QueryResponse
            _patch_query_endpoint(route)

# ===================================================================
# Internal helpers
# ===================================================================


def _build_chunks_dict_patched(
    chunking_result: list[dict], *, doc_id: str, file_path: str,
    blocks_path: str | None = None,
) -> dict[str, dict]:
    if blocks_path is None:
        blocks_path = _blocks_path_cv.get()
    chunks = _orig_build_chunks_dict(
        chunking_result, doc_id=doc_id, file_path=file_path
    )
    if not blocks_path or not chunks:
        return chunks
    from .sidecar import enrich_chunks_with_positions
    enrich_chunks_with_positions(chunks, chunking_result, blocks_path)
    return chunks


def _patch_pipeline(lpipe_module: Any) -> None:
    PipelineMixin = lpipe_module._PipelineMixin
    _save("process_single_document", PipelineMixin, "process_single_document")
    orig = PipelineMixin.process_single_document

    async def _wrapper(self, *, doc_id, status_doc, parsed_data, ctx):
        bp = str(parsed_data.get("blocks_path") or "").strip()
        _blocks_path_cv.set(bp if bp else None)
        try:
            return await orig(self, doc_id=doc_id, status_doc=status_doc,
                              parsed_data=parsed_data, ctx=ctx)
        finally:
            _blocks_path_cv.set(None)

    PipelineMixin.process_single_document = _wrapper

    _save(
        "pipeline_build_chunks_dict_from_chunking_result",
        lpipe_module,
        "build_chunks_dict_from_chunking_result",
    )
    lpipe_module.build_chunks_dict_from_chunking_result = (
        _build_chunks_dict_patched
    )


def _patch_query_param(lb_module: Any) -> None:
    _save("QueryParam", lb_module, "QueryParam")

    @dataclasses.dataclass
    class _EDWQueryParam(lb_module.QueryParam):  # type: ignore[valid-type]
        include_citation_highlights: bool = False

    lb_module.QueryParam = _EDWQueryParam


def _patch_request_model(qr_module: Any) -> None:
    from pydantic import Field
    _save("QueryRequest", qr_module, "QueryRequest")
    Base = qr_module.QueryRequest

    class _EDWRequest(Base):  # type: ignore[valid-type,misc]
        include_citation_highlights: bool = Field(
            default=False,
            description="When True, returns citation_highlights with "
                        "per-chunk bbox/page positions.")

    qr_module.QueryRequest = _EDWRequest


def _patch_response_model(qr_module: Any) -> None:
    from pydantic import Field
    _save("QueryResponse", qr_module, "QueryResponse")
    Base = qr_module.QueryResponse

    class _EDWResponse(Base):  # type: ignore[valid-type,misc]
        citation_highlights: dict | None = Field(
            default=None,
            description="Per-source chunk-level bbox/page highlights.")

    qr_module.QueryResponse = _EDWResponse


def _patch_query_endpoint(route: Any) -> None:
    if getattr(route, "_edw_patched", False):
        return
    setattr(route, "_edw_patched", True)

    original_endpoint = route.endpoint

    async def _wrapped(request):
        response = await original_endpoint(request)
        highlights = _citation_cv.get()
        if highlights and hasattr(response, "citation_highlights"):
            response.citation_highlights = highlights
            _citation_cv.set(None)
        return response

    route.endpoint = _wrapped
    if hasattr(route, "dependant") and route.dependant is not None:
        route.dependant.call = _wrapped
