"""
Query-chain patches -- propagate chunk-level positions through retrieval.

Each function here shadows the corresponding upstream LightRAG function.
Call ``apply_edw_rag_patches()`` from ``edw_rag_patches/__init__.py`` to
install them.

What changes
------------
- ``get_vector_context`` -- extracts ``positions`` from VDB results alongside content.
- ``merge_all_chunks`` -- preserves ``positions`` through round-robin merge + dedup.
- ``evidence_targets_from_chunk`` -- replaces chunk-level citation targets
  with canonical parser-block text for the LLM.
- ``convert_to_user_format`` -- detects positions on chunks, builds the
  ``citation_highlights`` sidecar inside ``data``.

The patches are **additive**: the existing ``file_path``-based reference
system continues to work unchanged.  ``positions`` is simply carried
alongside every other field and only used when present.
"""

from __future__ import annotations

from typing import Any

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    QueryParam,
)
from lightrag.utils import logger


def evidence_targets_from_chunk(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Return canonical parser blocks, not ambiguous chunk provenance IDs."""
    from .sidecar import build_evidence_targets

    return build_evidence_targets(chunk)


# ---------------------------------------------------------------------------
# 1. get_vector_context -- extract positions from VDB results
# ---------------------------------------------------------------------------

async def get_vector_context(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """Shadow the upstream ``_get_vector_context``, also extracting the
    ``positions`` field from VDB results.

    The upstream function builds a chunk dict with only ``content``,
    ``created_at``, ``file_path``, ``source_type``, and ``chunk_id``.
    We add ``positions`` from the VDB meta, so downstream stages
    (``_merge_all_chunks``, ``convert_to_user_format``) can use it.

    Why this matters
    ----------------
    The vector-DB stores whatever metadata was upserted with the chunk.
    Our patched ``build_chunks_dict_from_chunking_result`` stores a
    ``positions`` key in the chunk entry, which survives into
    ``chunks_vdb``.  This function simply reads it back out.
    """
    try:
        search_top_k = query_param.chunk_top_k or query_param.top_k
        results = await chunks_vdb.query(
            query, top_k=search_top_k, query_embedding=query_embedding,
        )
        if not results:
            return []

        valid_chunks = []
        for result in results:
            if "content" in result:
                chunk_with_metadata = {
                    "content": result["content"],
                    "created_at": result.get("created_at", None),
                    "file_path": result.get("file_path", "unknown_source"),
                    "source_type": "vector",
                    "chunk_id": result.get("id"),
                    # --- The one extra field ---
                    "positions": result.get("positions"),
                }
                valid_chunks.append(chunk_with_metadata)

        return valid_chunks

    except Exception as e:
        logger.error(f"Error in get_vector_context: {e}")
        return []


# ---------------------------------------------------------------------------
# 2. merge_all_chunks -- preserve positions through round-robin merge
# ---------------------------------------------------------------------------

async def merge_all_chunks(
    filtered_entities: list[dict],
    filtered_relations: list[dict],
    vector_chunks: list[dict],
    query: str = "",
    knowledge_graph_inst: BaseGraphStorage | None = None,
    text_chunks_db: BaseKVStorage | None = None,
    query_param: QueryParam | None = None,
    chunks_vdb: BaseVectorStorage | None = None,
    chunk_tracking: dict | None = None,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """Shadow the upstream ``_merge_all_chunks``, preserving the
    ``positions`` field through merge and deduplication.

    The upstream creates fresh ``{content, file_path, chunk_id}`` dicts
    for merged entries -- discarding every other field.  We add
    ``positions`` back so the sidecar builder can find them later.

    Merge strategy (unchanged from upstream):
    - Round-robin over three sources: vector_chunks, entity_chunks,
      relation_chunks.
    - Dedup by ``chunk_id`` (first source wins).
    - Backfill ``content_headings`` if the config flag is set.
    """
    import lightrag.operate as lo

    # Retrieve entity-related chunks from the text-chunks KV store.
    # These come from ``_find_related_text_unit_from_entities`` which
    # fetches the full stored dict (including any ``positions`` field).
    entity_chunks = []
    if filtered_entities and text_chunks_db:
        entity_chunks = await lo._find_related_text_unit_from_entities(
            filtered_entities,
            query_param,
            text_chunks_db,
            knowledge_graph_inst,
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Same for relation-related chunks.
    relation_chunks = []
    if filtered_relations and text_chunks_db:
        relation_chunks = await lo._find_related_text_unit_from_relations(
            filtered_relations,
            query_param,
            text_chunks_db,
            entity_chunks,
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Round-robin merge with dedup.  The only difference from the upstream
    # is carrying ``positions`` through.
    merged_chunks = []
    seen_chunk_ids = set()
    max_len = max(len(vector_chunks), len(entity_chunks), len(relation_chunks))

    for i in range(max_len):
        for source in [vector_chunks, entity_chunks, relation_chunks]:
            if i < len(source):
                chunk = source[i]
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    merged_chunks.append({
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                        "positions": chunk.get("positions"),  # ← preserved
                    })

    # Backfill durable provenance for every retrieved chunk. Headings are
    # optional inside the helper, but block evidence cannot depend on the
    # heading-display feature flag.
    if text_chunks_db:
        await lo._attach_content_headings(merged_chunks, text_chunks_db)

    return merged_chunks


# ---------------------------------------------------------------------------
# 3. convert_to_user_format -- builds citation_highlights sidecar
# ---------------------------------------------------------------------------

def convert_to_user_format(
    entities_context: list[dict],
    relations_context: list[dict],
    chunks: list[dict],
    references: list[dict],
    query_mode: str,
    entity_id_to_original: dict | None = None,
    relation_id_to_original: dict | None = None,
) -> dict[str, Any]:
    """Shadow the upstream ``convert_to_user_format``, additionally
    building a ``citation_highlights`` key in the ``data`` dict when
    any chunk carries ``positions``.

    The upstream produces::

        data = {
            "entities": [...],
            "relationships": [...],
            "chunks": [...],
            "references": [...],
        }

    We add::

        data["citation_highlights"] = {
            "version": 1,
            "sources": {
                "doc.pdf": {
                    "file_path": "doc.pdf",
                    "reference_id": "1",
                    "chunks": [
                        {"chunk_id": "...", "reference_id": "1",
                         "highlights": [{"page": 1, "bbox": {...}}]}
                    ]
                }
            }
        }

    This is called inside ``_build_context_str`` / ``_build_query_context``
    so the sidecar naturally flows into ``result["data"]``.
    """
    from edw_rag_patches import _originals
    upstream = _originals.get("convert_to_user_format")
    if upstream is None:
        from lightrag.utils import convert_to_user_format as upstream
    from .sidecar import build_citation_highlights

    # Let the upstream build the standard response first.
    result = upstream(
        entities_context,
        relations_context,
        chunks,
        references,
        query_mode,
        entity_id_to_original=entity_id_to_original,
        relation_id_to_original=relation_id_to_original,
    )

    data = result.get("data", {})
    has_positions = False

    if data and "chunks" in data:
        formatted_chunks = data["chunks"]
        has_positions = False

        chunk_by_id = {}
        for c in chunks:
            if isinstance(c, dict):
                cid = c.get("chunk_id") or c.get("id") or c.get("_id")
                if cid:
                    chunk_by_id[cid] = c

        for i, formatted_c in enumerate(formatted_chunks):
            orig_c = None
            cid = formatted_c.get("chunk_id") or formatted_c.get("id") or formatted_c.get("_id")
            if cid and cid in chunk_by_id:
                orig_c = chunk_by_id[cid]
            elif i < len(chunks) and isinstance(chunks[i], dict):
                orig_c = chunks[i]

            if orig_c:
                pos = orig_c.get("positions")
                ref_id = orig_c.get("reference_id")
                orig_cid = orig_c.get("chunk_id") or orig_c.get("id") or orig_c.get("_id")
                from lightrag.utils import logger
                logger.debug(f"[edw-rag] convert_to_user_format chunk {i}: "
                            f"cid={cid!r}, orig_cid={orig_cid!r}, "
                            f"has_positions={pos is not None}, ref_id={ref_id!r}")
                if pos:
                    formatted_c["positions"] = pos
                    has_positions = True
        if has_positions:
            highlights = build_citation_highlights(
                chunks=formatted_chunks,
                references=data.get("references", []),
            )
            if highlights.get("sources"):
                data["citation_highlights"] = highlights
                # Also build the citations map (chunk text + page evidence)
                from .sidecar import build_citations_map

                citations = build_citations_map(
                    chunks=formatted_chunks,
                    references=data.get("references", []),
                    citation_highlights=highlights,
                )
                if citations:
                    data["citations"] = citations
        else:
            # Even without positions, try building citations from chunk text alone
            from .sidecar import build_citations_map

            citations = build_citations_map(
                chunks=formatted_chunks,
                references=data.get("references", []),
            )
            if citations:
                data["citations"] = citations
    from lightrag.utils import logger
    logger.debug(f"[edw-rag] convert_to_user_format done: has_positions={has_positions}, "
                 f"citations_in_data={'citations' in data}, "
                 f"highlights_in_data={'citation_highlights' in data}")
    return result
