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

def _generate_ref_list_per_chunk(
    chunks: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Assign a unique reference_id per content box (chunk)."""
    if not chunks:
        return [], []
    ref_list: list[dict] = []
    updated: list[dict] = []
    for i, chunk in enumerate(chunks):
        c = chunk.copy()
        rid = chunk.get("chunk_id") or str(i + 1)
        c["reference_id"] = rid
        updated.append(c)
        fp = c.get("file_path", "")
        content = (c.get("content") or "").strip()
        label = content[:40].replace("\n", " ")
        if len(content) > 40:
            label += "…"
        ref_list.append({
            "reference_id": rid,
            "file_path": fp,
            "label": label,
        })
    return ref_list, updated


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
    # -- 1c. Assign per-chunk citation IDs instead of per-file -----------
    _save("generate_reference_list_from_chunks", lu, "generate_reference_list_from_chunks")
    lu.generate_reference_list_from_chunks = _generate_ref_list_per_chunk
    _save("operate_generate_reference_list_from_chunks", lo, "generate_reference_list_from_chunks")
    lo.generate_reference_list_from_chunks = _generate_ref_list_per_chunk
    # -- 2. Query pipeline: propagate positions through retrieval ----------
    from . import query_chain as qc

    _save("_get_vector_context", lo, "_get_vector_context")
    lo._get_vector_context = qc.get_vector_context

    _save("_merge_all_chunks", lo, "_merge_all_chunks")
    lo._merge_all_chunks = qc.merge_all_chunks

    _save("_citation_targets_from_chunk", lo, "_citation_targets_from_chunk")
    lo._citation_targets_from_chunk = qc.evidence_targets_from_chunk

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

    # -- 5c. Optional native Qdrant dense + BM25 hybrid retrieval ---------
    from . import qdrant_hybrid

    if qdrant_hybrid.enabled():
        import lightrag.kg.qdrant_impl as qdrant_impl

        _originals["qdrant_hybrid"] = (
            qdrant_impl.QdrantVectorDBStorage,
            qdrant_hybrid.apply_qdrant_hybrid_patch(
                qdrant_impl.QdrantVectorDBStorage,
                qdrant_impl.models,
                lu.logger,
            ),
        )

    # -- 6. Patch aquery_llm to bubble citation_highlights up --------------
    _save("aquery_llm", ll.LightRAG, "aquery_llm")
    _orig_aql = ll.LightRAG.aquery_llm

    async def _aql_wrapper(self, query: str, param=None, system_prompt=None, **kwargs):
        if system_prompt is not None:
            kwargs["system_prompt"] = system_prompt
        from lightrag.utils import logger
        from .sidecar import (
            build_evidence_map,
            filter_to_cited,
            parse_citation_selectors,
            parse_cited_ids,
            references_from_evidence,
            render_structured_answer,
        )

        result = await _orig_aql(self, query, param=param, **kwargs)
        data = result.get("data", {}) if isinstance(result, dict) else {}

        has_ch = isinstance(data, dict) and "citation_highlights" in data

        llm_content = (result.get("llm_response") or {}).get("content") or ""
        cited_ids = set()
        citation_selectors: list[dict[str, str]] = []
        evidence_map: dict[str, dict[str, Any]] = {}

        if llm_content:
            evidence_map = build_evidence_map(data.get("chunks") or [])
            structured_answer = render_structured_answer(llm_content, evidence_map)
            if not structured_answer:
                logger.error(
                    "[edw-rag] rejected non-structured LLM answer or unsent evidence ID"
                )
                llm_content = (
                    "Sorry, I couldn't validate a grounded answer for this query."
                )
                data["structured_answer_error"] = "invalid_model_evidence"
            else:
                llm_content, answer_segments = structured_answer
                data["answer_segments"] = answer_segments
                cited_ids = parse_cited_ids(llm_content)
                citation_selectors = parse_citation_selectors(llm_content)

            if "llm_response" in result and isinstance(result["llm_response"], dict):
                result["llm_response"]["content"] = llm_content

        # Response provenance must describe the answer, not every chunk
        # considered during retrieval. This also leaves ``references`` empty
        # when the model supplied no inline citation at all.
        if has_ch:
            ch, cit, refs, cks = filter_to_cited(
                data.get("citation_highlights"),
                data.get("citations"),
                data.get("references"),
                data.get("chunks"),
                cited_ids,
                citation_selectors,
            )
            data["citation_highlights"] = ch
            data["citations"] = cit
            data["references"] = refs
            data["chunks"] = cks
            logger.debug(f"[edw-rag] filtered to {len(cited_ids)} cited box ids")

        if llm_content:
            # This is authoritative even when no PDF positions were stored:
            # references describe blocks selected by the model, never the
            # broader retrieval chunk set.
            data["references"] = references_from_evidence(
                citation_selectors, evidence_map
            )

        if isinstance(data, dict) and "citation_highlights" in data:
            _citation_cv.set(data["citation_highlights"])
        return result
    ll.LightRAG.aquery_llm = _aql_wrapper

    # -- 7. Patch prompts for inline citation markers ----------------------
    import lightrag.prompt as lp

    for key in ("rag_response", "naive_rag_response"):
        _save(f"prompt_{key}", lp.PROMPTS, key)
        tmpl = lp.PROMPTS[key]

        structured_marker = "EDW structured evidence response"
        structured_instruction = (
            "\n  - EDW structured evidence response: This replaces every inline-citation "
            "and references-format instruction above. Return ONLY a valid JSON object; "
            "do not use Markdown fences or inline citation markers. Its exact shape is "
            '`{{"segments":[{{"markdown":"answer text","evidence_ids":["e_abc"]}}]}}`.\n'
            "  - Each segment is one claim or tightly related supported passage. "
            "Its `evidence_ids` must contain only IDs from the supplied `evidence` "
            "objects (called `citation_targets` in naive mode); each is canonical "
            "block evidence. The surrounding chunk `content`, if present, is "
            "retrieval-only routing data and must not be used as evidence; use only "
            "an evidence object's `text`. "
            "Do not emit document IDs, chunk IDs, paths, block "
            "IDs, page numbers, or citation syntax.\n"
            "  - Cite every factual segment with one or more evidence IDs. For a "
            "no-context response only, use an empty `evidence_ids` array.\n"
            "  - The `markdown` string may use normal Markdown but must not contain "
            "`[^`. The server validates evidence IDs and adds citations itself.\n"
        )
        if structured_marker not in tmpl:
            structured_instruction = structured_instruction.replace(
                "EDW structured evidence response", structured_marker
            )
            tmpl = tmpl.replace(
                "---Context---", structured_instruction + "\n---Context---"
            )

        old_ref = (
            "4. References Section Format:\n"
            "  - The References section should be under heading: `### References`\n"
            '  - Reference list entries should adhere to the format: `* [n] Document Title`. '
            "Do not include a caret (`^`) after opening square bracket (`[`).\n"
            "  - The Document Title in the citation must retain its original language.\n"
            "  - Output each citation on an individual line\n"
            "  - Provide maximum of 5 most relevant citations.\n"
            "  - Do not generate footnotes section or any comment, summary, or explanation after the references.\n"
        )
        new_ref = (
            "4. Citations and References Format:\n"
            "  - INLINE FORMAT: Cite your sources inline immediately after the claim, fact, or synthesized idea using this exact format: [^reference_id]\n"
            "    - Example: Operating costs dropped significantly [^doc-XXX-chunk-000].\n"
            "  - PDF HIGHLIGHTING (DIRECT CITATIONS): When you directly quote exact words or phrases from the PDF context, you must wrap the exact source text in <mark> tags, immediately followed by the citation format. The text inside the <mark> tags MUST be an exact substring match of the source text.\n"
            "    - Example: The report states that <mark>net retention remained at 110%</mark> [^doc-XXX-chunk-000].\n"
            "  - Only cite boxes whose information you DIRECTLY used to support a fact.\n"
            "  - The References section should be under heading: `### References`\n"
            "  - Reference list entries should adhere to the format: `- [reference_id] Document Title`.\n"
            "  - Output each citation on an individual line.\n"
        )
        if old_ref in tmpl:
            tmpl = tmpl.replace(old_ref, new_ref)

        old_ex = (
            "5. Reference Section Example:\n"
            "```\n"
            "### References\n"
            "\n"
            "- [1] Document Title One\n"
            "- [2] Document Title Two\n"
            "- [3] Document Title Three\n"
            "```\n"
        )
        new_ex = (
            "5. Citation and Reference Example:\n"
            "```\n"
            "Amazon expanded EV charging in India[^doc-1-chunk-0] "
            "through <mark>The Climate Pledge</mark> [^doc-2-chunk-1].\n\n"
            "### References\n\n"
            "- [doc-1-chunk-0] Climate Pledge net-zero\n"
            "- [doc-2-chunk-1] Renewable energy by 2030\n"
            "```\n"
        )
        if old_ex in tmpl:
            tmpl = tmpl.replace(old_ex, new_ex)

        lp.PROMPTS[key] = tmpl

    # -- 9. Patch reference list to use [^N] instead of [N] ----------------
    _save("_build_context_str", lo, "_build_context_str")
    _orig_build_context_str = lo._build_context_str

    async def _build_context_str_wrapper(*args, **kwargs):
        result = await _orig_build_context_str(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 1:
            from .sidecar import build_evidence_map
            import json

            raw_data = result[1] if len(result) > 1 else {}
            data = raw_data.get("data", {}) if isinstance(raw_data, dict) else {}
            evidence_map = build_evidence_map(data.get("chunks") or [])
            if evidence_map:
                # Do not show retrieval chunks, graph descriptions, file paths,
                # or parser IDs to the LLM. It reasons only over canonical block
                # text and selects compact, server-validated evidence IDs.
                public_evidence = [
                    {"evidence_id": item["evidence_id"], "text": item["text"]}
                    for item in evidence_map.values()
                ]
                ctx = (
                    "Canonical Evidence Blocks (the only factual source for the answer):\n"
                    "```json\n"
                    + json.dumps(public_evidence, ensure_ascii=False)
                    + "\n```"
                )
                result = (ctx,) + result[1:]
        return result

    lo._build_context_str = _build_context_str_wrapper


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
        "_citation_targets_from_chunk": (lo, "_citation_targets_from_chunk"),
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
    qdrant_hybrid = _originals.pop("qdrant_hybrid", None)
    if qdrant_hybrid:
        storage_cls, originals = qdrant_hybrid
        storage_cls.initialize = originals["initialize"]
        storage_cls.upsert = originals["upsert"]
        storage_cls.delete = originals["delete"]
        storage_cls._flush_pending_vector_ops = originals["flush"]
        storage_cls.query = originals["query"]


def patch_app_routes(app: Any) -> None:
    """Patch FastAPI app routes to include citation_highlights in /query."""
    import lightrag.api.routers.query_routes as qr
    from lightrag.utils import logger

    patched = 0
    for route in app.routes:
        if not hasattr(route, "methods") or not hasattr(route, "endpoint"):
            continue
        path = getattr(route, "path", "")
        matched = path.rstrip("/").endswith(("/query", "/query/stream"))
        if matched and "POST" in route.methods:
            if hasattr(route, "response_model") and route.response_model is not None:
                route.response_model = qr.QueryResponse
            _patch_query_endpoint(route)
            patched += 1
            logger.debug(f"[edw-rag] patched route: {path}")
    logger.debug(f"[edw-rag] patch_app_routes: total routes={len(app.routes)}, patched={patched}")

# ===================================================================
# Internal helpers
# ===================================================================


def _build_chunks_dict_patched(
    chunking_result: list[dict], *, doc_id: str, file_path: str,
    blocks_path: str | None = None,
) -> dict[str, dict]:
    if blocks_path is None:
        blocks_path = _blocks_path_cv.get()
    from lightrag.utils import logger
    logger.debug(f"[edw-rag] _build_chunks_dict_patched: blocks_path={blocks_path!r}, "
                 f"chunks_in_result={len(chunking_result) if chunking_result else 0}")
    chunks = _orig_build_chunks_dict(
        chunking_result, doc_id=doc_id, file_path=file_path
    )
    if not blocks_path:
        logger.debug(f"[edw-rag] blocks_path empty or None — skipping position enrichment")
        return chunks
    if not chunks:
        logger.debug(f"[edw-rag] no chunks produced — skipping position enrichment")
        return chunks
    from .sidecar import enrich_chunks_with_positions
    from .sidecar import _load_blocks_jsonl
    blocks = _load_blocks_jsonl(blocks_path)
    logger.debug(f"[edw-rag] blocks loaded: {len(blocks) if blocks else 0} from {blocks_path}")
    enrich_chunks_with_positions(chunks, chunking_result, blocks_path)
    # Check if any chunks got positions
    pos_count = sum(1 for c in chunks.values() if c.get("positions"))
    logger.debug(f"[edw-rag] chunks with positions after enrichment: {pos_count}/{len(chunks)}")
    return chunks


def _patch_pipeline(lpipe_module: Any) -> None:
    PipelineMixin = lpipe_module._PipelineMixin
    _save("process_single_document", PipelineMixin, "process_single_document")
    orig = PipelineMixin.process_single_document

    async def _wrapper(self, *, doc_id, status_doc, parsed_data, ctx):
        bp = str(parsed_data.get("blocks_path") or "").strip()
        from lightrag.utils import logger
        logger.debug(f"[edw-rag] pipeline _wrapper: blocks_path={bp!r}")
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
    from edw_rag_patches import _citation_cv
    _save("QueryResponse", qr_module, "QueryResponse")
    Base = qr_module.QueryResponse

    class _EDWResponse(Base):  # type: ignore[valid-type,misc]
        citation_highlights: dict | None = Field(
            default_factory=lambda: _citation_cv.get(),
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
        from lightrag.utils import logger
        logger.debug(f"[edw-rag] endpoint wrapper: highlights={'found' if highlights else 'None'}, "
                     f"response_has_attr={hasattr(response, 'citation_highlights')}")
        if highlights and hasattr(response, "citation_highlights"):
            response.citation_highlights = highlights
            _citation_cv.set(None)
        return response

    route.endpoint = _wrapped
    if hasattr(route, "dependant") and route.dependant is not None:
        route.dependant.call = _wrapped
