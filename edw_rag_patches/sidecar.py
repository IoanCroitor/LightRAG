"""
Chunk-level citation highlight builder for EDW-RAG.

Core logic: maps chunks back to Docling block positions from
``.parsed/blocks.jsonl``, producing per-chunk bounding-box position
arrays for a custom PDF viewer.

Strategy overview
-----------------
The enrichment is called **at chunk-insertion time** (inside
``build_chunks_dict_from_chunking_result``), so positions are stored
directly in the vector-DB meta alongside chunk content.  No secondary
lookup needed at query time.

Two mapping strategies are used, depending on which chunker produced
the output:

**P‑chunker** (paragraph_semantic)
    The chunk output carries a ``sidecar.refs`` list of block IDs from
    ``blocks.jsonl``.  We look up each block ID in the file, collect
    every ``IRPosition`` with ``type="bbox"``, and normalise them into
    compact highlight dicts.

**F / R / V chunkers** (token_size, recursive_character, semantic_vector)
    These chunkers produce ``_source_span`` -- character offsets into the
    merged text -- but no block IDs.  We fall back to **text-overlap
    matching**: for each chunk we find blocks whose content shares the
    most text with the chunk, and take their positions.  This is
    approximate but reliable for verbatim chunking.

Both paths produce the same output format, so the query pipeline never
needs to know which chunker was used.
"""

from __future__ import annotations

import hashlib
import re

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def enrich_chunks_with_positions(
    chunks: dict[str, dict[str, Any]],
    chunking_result: list[dict[str, Any]],
    blocks_path: str,
) -> None:
    """Mutate ``chunks`` (the dict from ``build_chunks_dict_from_chunking_result``)
    in place, adding a ``positions`` key to each entry whose content can be
    traced to block positions in ``blocks.jsonl``.

    Parameters
    ----------
    chunks:
        The return value of ``build_chunks_dict_from_chunking_result``
        -- a ``{chunk_key: {content, full_doc_id, file_path, ...}}`` dict.
        Mutated in place.
    chunking_result:
        The raw chunker output (from P, F, R, or V) **before** it went
        through ``build_chunks_dict_from_chunking_result``.  We need the
        original ``sidecar.refs`` (P) and ``_source_span`` (F/R/V).
    blocks_path:
        Filesystem path to the ``.blocks.jsonl`` file inside the
        ``.parsed/`` sidecar directory.
    """
    blocks = _load_blocks_jsonl(blocks_path)
    if not blocks:
        return

    # --- Index blocks for fast lookup -----------------------------------

    # block_by_content: maps block text -> list of raw IR position dicts.
    # Used for text-overlap matching (F/R/V fallback).
    # Multiple blocks can have identical content (e.g. repeated table rows),
    # so values are lists, not single entries.
    block_by_content: dict[str, list[dict]] = {}

    # block_by_id: maps blockid -> full block dict.
    # Used for P-chunker's direct block-ID lookup.
    block_by_id: dict[str, dict] = {}

    for blk in blocks:
        blk_content = blk.get("content", "")
        blk_positions = blk.get("positions") or []
        if blk_content:
            block_by_content.setdefault(blk_content, []).extend(blk_positions)
        bid = blk.get("blockid")
        if bid:
            block_by_id[bid] = blk

    # --- Map chunk_order_index to the final chunk_key -------------------
    # The chunking_result entries are indexed by ``chunk_order_index``,
    # but the ``chunks`` dict uses opaque keys like ``doc-xxx-chunk-000``.
    # We build a reverse mapping to bridge them.
    order_to_key: dict[int, str] = {}
    for ck, cv in chunks.items():
        oi = cv.get("chunk_order_index")
        if isinstance(oi, int):
            order_to_key[oi] = ck

    # --- For each chunk in chunking_result, collect positions ----------
    chunk_positions: dict[str, list[dict]] = {}

    for chunk_dp in chunking_result:
        chunk_content = chunk_dp.get("content", "")
        if not chunk_content:
            continue
        oi = chunk_dp.get("chunk_order_index")
        if not isinstance(oi, int):
            continue
        chunk_key = order_to_key.get(oi)
        if not chunk_key:
            continue

        positions = _get_chunk_positions(
            chunk_dp, chunk_content, block_by_id, block_by_content,
        )
        if positions:
            chunk_positions[chunk_key] = positions

    # --- Write positions back into the chunks dict ----------------------
    for ck, positions in chunk_positions.items():
        if positions:
            chunks[ck]["positions"] = positions


def _get_chunk_positions(
    chunk_dp: dict[str, Any],
    chunk_content: str,
    block_by_id: dict[str, dict],
    block_by_content: dict[str, list[dict]],
) -> list[dict]:
    """Resolve the positions for a single chunk entry.

    Tries three strategies in order of decreasing specificity:

    1. **P-chunker** -- ``sidecar.refs`` block IDs (direct, exact)
    2. **F/R/V** -- ``_source_span`` char-offset overlap (approximate)
    3. **Text-overlap fallback** -- shared text between chunk and blocks

    Positions are deduplicated at the end (same page + bbox = same highlight).
    """
    positions: list[dict] = []

    # --- Strategy 1: P-chunker -- sidecar.refs block IDs ----------------
    sidecar = chunk_dp.get("sidecar")
    if isinstance(sidecar, dict):
        refs = sidecar.get("refs")
        if isinstance(refs, list):
            for ref in refs:
                if isinstance(ref, dict):
                    bid = ref.get("id")
                    if bid and bid in block_by_id:
                        blk = block_by_id[bid]
                        positions.extend(
                            _normalize_positions(
                                blk.get("positions") or [], block_id=str(bid)
                            )
                        )

    # --- Strategy 2: F/R/V -- _source_span char-offset overlap ----------
    # ``_source_span`` gives the character range within the merged text
    # stored in ``full_docs``.  We attempt to find blocks whose content
    # falls within this range.
    source_span = chunk_dp.get("_source_span")
    if not positions and isinstance(source_span, dict):
        chunk_start = source_span.get("start")
        chunk_end = source_span.get("end")
        if chunk_start is not None and chunk_end is not None:
            positions = _positions_by_char_span(
                chunk_start, chunk_end, block_by_content,
            )

    # --- Strategy 3: text-overlap fallback (all chunkers) --------------
    if not positions:
        positions = _positions_by_text_overlap(chunk_content, block_by_content)

    # --- Deduplicate: same page + same bbox = one highlight ------------
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for pos in positions:
        key = (
            pos.get("page"),
            json.dumps(pos.get("bbox"), sort_keys=True),
            pos.get("origin"),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(pos)

    return deduped


def _positions_by_char_span(
    chunk_start: int,
    chunk_end: int,
    block_by_content: dict[str, list[dict]],
) -> list[dict]:
    """Map char offsets in merged text to block positions (DEPRECATED).

    This is an approximate fallback.  Merged text is all block contents
    concatenated with double-newlines.  Since we don't have the block
    ordering here, we return all positions from every block.  The
    text-overlap fallback (strategy 3) is more reliable.
    """
    positions: list[dict] = []
    for blk_content, blk_positions in block_by_content.items():
        positions.extend(_normalize_positions(blk_positions))
    return positions


def _positions_by_text_overlap(
    chunk_content: str,
    block_by_content: dict[str, list[dict]],
) -> list[dict]:
    """Find block positions via text overlap.

    For each block, we measure how much text it shares with the chunk
    (in characters).  Blocks sharing at least 20 characters contribute
    their positions.  This works reliably for all chunker types because:

    - Chunks are verbatim substrings of the original document text.
    - Block content in blocks.jsonl is also verbatim from the document.
    - Even after markdown formatting (P-chunker), core content words survive.
    """
    if not chunk_content or not block_by_content:
        return []

    positions: list[dict] = []
    chunk_stripped = chunk_content.strip()
    if not chunk_stripped:
        return []

    # Fast path: exact match (common when a chunk covers exactly one block)
    if chunk_stripped in block_by_content:
        return _normalize_positions(block_by_content[chunk_stripped])

    # Slow path: find blocks with significant text overlap
    for blk_content, blk_positions in block_by_content.items():
        if not blk_content.strip():
            continue
        shared = _shared_text_length(chunk_stripped, blk_content.strip())
        if shared > 20:  # at least 20 characters of overlap
            positions.extend(_normalize_positions(blk_positions))

    return positions


def _shared_text_length(a: str, b: str) -> int:
    """Estimate how much text two strings share (in characters).

    For short strings (≤200 chars) we test substring containment directly.
    For longer strings we fall back to word-set intersection length,
    which is O(n) in the number of words and avoids the O(n²) cost of
    longest-common-substring computation.
    """
    if len(a) <= 200 or len(b) <= 200:
        if a in b or b in a:
            return min(len(a), len(b))
    words_a = set(a.split())
    words_b = set(b.split())
    common = words_a & words_b
    return sum(len(w) for w in common)


def _normalize_positions(
    raw_positions: list[dict], *, block_id: str | None = None
) -> list[dict]:
    """Convert IR (intermediate-representation) position dicts as stored
    in ``blocks.jsonl`` into the compact citation-highlight format.

    Input format (from blocks.jsonl)::

        {"type": "bbox", "anchor": "1",
         "range": [72.0, 144.0, 523.0, 160.0],
         "origin": "LEFTTOP"}

    Output format::

        {"page": 1,
         "bbox": {"l": 72.0, "t": 144.0, "r": 523.0, "b": 160.0},
         "origin": "LEFTTOP"}

    Notes
    -----
    - Only ``type == "bbox"`` entries are converted; other position types
      (``paraid``, ``heading``, ``absolute``) are silently skipped.
    - The ``anchor`` field is the page number (as a string); we parse it
      as an integer.  Non-numeric anchors produce ``page = None`` and the
      entry is still included -- the viewer can decide how to handle it.
    - The ``range`` array must have exactly 4 elements (l, t, r, b);
      anything else produces no ``bbox`` key.
    """
    highlights: list[dict] = []
    for pos in raw_positions:
        if not isinstance(pos, dict):
            continue
        if pos.get("type") != "bbox":
            # Skip paragraph-anchor, heading, and absolute position types
            continue

        # Page number from the anchor field
        anchor = pos.get("anchor")
        try:
            page = int(anchor) if anchor is not None else None
        except (ValueError, TypeError):
            page = None

        # Bounding box coordinates
        bbox_range = pos.get("range")
        bbox: dict | None = None
        if isinstance(bbox_range, (list, tuple)) and len(bbox_range) == 4:
            bbox = {
                "l": float(bbox_range[0]),
                "t": float(bbox_range[1]),
                "r": float(bbox_range[2]),
                "b": float(bbox_range[3]),
            }

        highlight: dict = {}
        if page is not None:
            highlight["page"] = page
        if bbox is not None:
            highlight["bbox"] = bbox
        origin = pos.get("origin")
        if origin:
            highlight["origin"] = origin  # "LEFTTOP" or "LEFTBOTTOM"
        if block_id:
            highlight["block_id"] = block_id

        if highlight:
            highlights.append(highlight)

    return highlights


def _load_blocks_jsonl(blocks_path: str) -> list[dict[str, Any]]:
    """Read ``type == "content"`` rows from a ``.blocks.jsonl`` file.

    The file contains one JSON object per line.  The first line is
    typically a ``type == "meta"`` row (carrying document-level metadata
    like ``bbox_attributes.origin``).  Subsequent rows are
    ``type == "content"`` blocks representing paragraphs, headings,
    tables, etc. -- each with a ``positions`` array.

    Returns an empty list if the file doesn't exist or can't be read.
    """
    path = Path(blocks_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("type") == "content":
                    rows.append(obj)
    except OSError:
        return []
    return rows


# ---------------------------------------------------------------------------
# Query-time evidence hydration
# ---------------------------------------------------------------------------

def build_evidence_targets(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve a retrieved chunk's provenance to canonical, citeable blocks.

    Chunks are useful for vector retrieval but can combine unrelated parser
    blocks. The LLM must therefore see the canonical block text paired with a
    compact evidence ID, rather than guessing which block inside chunk content
    supports a claim.
    """
    document_id = str(chunk.get("document_id") or "").strip()
    file_path = str(chunk.get("file_path") or "").strip()
    sidecar = chunk.get("sidecar")
    if not document_id or not file_path or not isinstance(sidecar, dict):
        return []

    refs = sidecar.get("refs")
    if not isinstance(refs, list):
        refs = [sidecar]
    blocks = _blocks_by_id(file_path)
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in refs:
        if not isinstance(ref, dict) or ref.get("type") != "block":
            continue
        block_id = str(ref.get("id") or "").strip()
        block = blocks.get(block_id)
        if not block or block_id in seen:
            continue
        seen.add(block_id)
        text = str(block.get("content") or "").strip()
        if not text:
            continue
        start = ref.get("start")
        end = ref.get("end")
        start = start if isinstance(start, int) and start >= 0 else 0
        end = end if isinstance(end, int) and end >= start else len(text)
        end = min(end, len(text))
        evidence_text = text[start:end].strip()
        if not evidence_text:
            continue
        targets.append(
            {
                "evidence_id": _evidence_id(document_id, block_id),
                "document_id": document_id,
                "file_path": file_path,
                "section_id": "root",
                "block_id": block_id,
                "text": evidence_text,
                "start": start,
                "end": end,
            }
        )
    return targets


def build_evidence_map(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the allowed evidence IDs for a query, deduplicated by block."""
    evidence: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for target in build_evidence_targets(chunk):
            evidence.setdefault(target["evidence_id"], target)
    return evidence


def references_from_evidence(
    selectors: list[dict[str, str]], evidence_map: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    """Build public references from the canonical blocks selected by the LLM."""
    references: list[dict[str, str]] = []
    seen_documents: set[str] = set()
    for selector in selectors:
        document_id = selector.get("chunk_id", "")
        block_id = selector.get("block_id", "")
        if document_id in seen_documents:
            continue
        evidence = next(
            (
                item
                for item in evidence_map.values()
                if item.get("document_id") == document_id
                and (not block_id or item.get("block_id") == block_id)
            ),
            None,
        )
        if not evidence:
            continue
        seen_documents.add(document_id)
        references.append(
            {
                "reference_id": document_id,
                "file_path": str(evidence["file_path"]),
            }
        )
    return references


def _blocks_by_id(file_path: str) -> dict[str, dict[str, Any]]:
    """Load a document's canonical parser blocks for one query-time lookup."""
    try:
        from lightrag.utils_pipeline import parsed_artifact_dir_for

        artifact_dir = parsed_artifact_dir_for(file_path)
        candidates = sorted(artifact_dir.glob("*.blocks.jsonl"))
    except Exception:
        return {}
    if not candidates:
        return {}
    blocks_path = candidates[0]
    try:
        mtime_ns = blocks_path.stat().st_mtime_ns
    except OSError:
        return {}
    return _cached_blocks_by_id(str(blocks_path), mtime_ns)


@lru_cache(maxsize=128)
def _cached_blocks_by_id(
    blocks_path: str, _mtime_ns: int
) -> dict[str, dict[str, Any]]:
    """Cache immutable parsed artifacts while invalidating after reprocessing."""
    return {
        str(block.get("blockid")): block
        for block in _load_blocks_jsonl(blocks_path)
        if block.get("blockid")
    }


def _evidence_id(document_id: str, block_id: str) -> str:
    """Create a compact deterministic ID that the LLM may select."""
    digest = hashlib.sha256(f"{document_id}\x1f{block_id}".encode()).hexdigest()
    return f"e_{digest[:12]}"


def render_structured_answer(
    content: str, evidence_map: dict[str, dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]] | None:
    """Validate the model's JSON answer and render server-owned citations.

    The model is allowed to select only an ``evidence_id`` it received in the
    prompt. Document IDs, parser block IDs, display indices, and inline
    citation syntax are constructed here, never accepted from the model.
    """
    raw = content.strip()
    if raw.startswith("```json") and raw.endswith("```"):
        raw = raw[7:-3].strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(segments, list) or not segments:
        return None

    normalized: list[dict[str, Any]] = []
    ordered_evidence: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            return None
        markdown = segment.get("markdown")
        evidence_ids = segment.get("evidence_ids")
        if (
            not isinstance(markdown, str)
            or not markdown.strip()
            or "[^" in markdown
            or not isinstance(evidence_ids, list)
            or not all(isinstance(evidence_id, str) for evidence_id in evidence_ids)
            or any(evidence_id not in evidence_map for evidence_id in evidence_ids)
        ):
            return None
        ids = list(dict.fromkeys(evidence_ids))
        normalized.append({"markdown": markdown.strip(), "evidence_ids": ids})
        for evidence_id in ids:
            if evidence_id not in ordered_evidence:
                ordered_evidence.append(evidence_id)

    display_index = {
        evidence_id: index + 1
        for index, evidence_id in enumerate(ordered_evidence)
    }
    rendered_segments: list[str] = []
    for segment in normalized:
        citations = "".join(
            _citation_marker(display_index[evidence_id], evidence_map[evidence_id])
            for evidence_id in segment["evidence_ids"]
        )
        rendered_segments.append(f"{segment['markdown']}{citations}")
    return "\n\n".join(rendered_segments), normalized


def _citation_marker(index: int, evidence: dict[str, Any]) -> str:
    return (
        f"[^{index}:{evidence['document_id']}§{evidence['section_id']}"
        f"¶{evidence['block_id']}]"
    )


# ---------------------------------------------------------------------------
# Sidecar JSON builder
# ---------------------------------------------------------------------------

def build_citation_highlights(
    chunks: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the ``citation_highlights`` sidecar from processed chunks.

    ``chunks`` must already carry ``reference_id`` and ``positions`` fields
    (added by ``enrich_chunks_with_positions`` during storage).

    The returned dict is structured as a **sidecar JSON** that a custom PDF
    viewer can consume directly::

        {
          "version": 1,
          "sources": {
            "/path/doc.pdf": {
              "file_path": "/path/doc.pdf",
              "reference_id": "1",
              "chunks": [
                {
                  "chunk_id": "chunk-abc123",
                  "reference_id": "1",
                  "highlights": [
                    {"page": 1,
                     "bbox": {"l": 72, "t": 144, "r": 523, "b": 160}},
                    {"page": 1,
                     "bbox": {"l": 72, "t": 200, "r": 480, "b": 215}}
                  ]
                }
              ]
            }
          }
        }

    Parameters
    ----------
    chunks:
        The ``chunks`` list from ``data["chunks"]``, after
        ``convert_to_user_format``.  Each entry should have
        ``reference_id``, ``chunk_id``, ``file_path``, and ``positions``.
    references:
        The ``references`` list from ``data["references"]``, used to
        map ``file_path`` to ``reference_id``.

    Returns
    -------
    dict
        Sidecar JSON with ``version`` and ``sources`` keys.
        Empty ``sources`` if no chunks have positions.
    """
    # Build file_path → reference_id map from the references list
    ref_map: dict[str, str] = {}
    for ref in references:
        if isinstance(ref, dict):
            fp = ref.get("file_path", "")
            rid = ref.get("reference_id", "")
            if fp and fp != "unknown_source":
                ref_map[fp] = rid

    sources: dict[str, dict] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        fp = chunk.get("file_path", "")
        if not fp or fp == "unknown_source":
            continue
        rid = chunk.get("reference_id", "")
        cid = chunk.get("chunk_id", "")
        chunk_positions = chunk.get("positions") or []
        if not chunk_positions:
            continue  # skip chunks without position data

        # Give every visual rectangle a stable ID. The block-relative range
        # identifies the cited source span; a per-block box ordinal keeps IDs
        # unique when one block wraps across several lines/columns.
        refs_by_block: dict[str, dict[str, Any]] = {}
        sidecar = chunk.get("sidecar")
        if isinstance(sidecar, dict):
            for ref in sidecar.get("refs") or []:
                if isinstance(ref, dict) and ref.get("type") == "block" and ref.get("id"):
                    refs_by_block[str(ref["id"])] = ref

        highlights: list[dict[str, Any]] = []
        box_ordinals: dict[str, int] = {}
        for position in chunk_positions:
            if not isinstance(position, dict):
                continue
            highlight = position.copy()
            block_id = str(highlight.get("block_id") or "")
            source_ref = refs_by_block.get(block_id, {})
            start = source_ref.get("start")
            end = source_ref.get("end")
            start = start if isinstance(start, int) and start >= 0 else 0
            end = end if isinstance(end, int) and end >= start else 0
            if block_id:
                highlight["block_id"] = block_id
                highlight["start"] = start
                highlight["end"] = end
                ordinal = box_ordinals.get(block_id, 0)
                box_ordinals[block_id] = ordinal + 1
                highlight["highlight_id"] = f"hl_{block_id}_{start}_{end}_{ordinal}"
            highlights.append(highlight)

        if fp not in sources:
            sources[fp] = {
                "file_path": fp,
                "reference_id": ref_map.get(fp, rid),
                "chunks": [],
            }

        sources[fp]["chunks"].append({
            "chunk_id": cid,
            "reference_id": rid,
            "highlights": highlights,
        })

    return {
        "version": 1,
        "sources": sources,
    }



def build_citations_map(
    chunks: list[dict[str, Any]],
    references: list[dict[str, Any]],
    citation_highlights: dict[str, Any] | None = None,
) -> dict[str, dict]:
    """Build a ``citations`` map for the query response.

    Groups chunk text + page evidence by ``reference_id`` so the frontend
    can show inline evidence cards when the user clicks ``[^N]``.

    Output shape::

        {
          "1": {
            "reference_id": "1",
            "file_path": "2024-amazon-sustainability-report_p16.pdf",
            "evidence": [
              {
                "chunk_id": "doc-xxx-chunk-000",
                "text": "Full chunk text…",
                "page": 1,
                "bbox": {"l": 70, "t": 125, "r": 490, "b": 205},
              }
            ]
          }
        }

    Parameters
    ----------
    chunks:
        The ``data["chunks"]`` list after ``convert_to_user_format``.
        Each entry should have ``reference_id``, ``chunk_id``, ``content``,
        and optionally ``file_path``.
    references:
        The ``data["references"]`` list used to key the output.
    citation_highlights:
        Optional pre-built ``citation_highlights`` dict.  When given, the
        page + bbox for each chunk is resolved from its highlights.

    Returns
    -------
    dict
        ``{reference_id: {reference_id, file_path, evidence: [...]}}``.
    """
    # Pre-index citation_highlights by chunk_id for fast lookup
    ch_by_chunk: dict[str, list[dict]] = {}
    if citation_highlights and isinstance(citation_highlights, dict):
        for src in (citation_highlights.get("sources") or {}).values():
            for ck in (src.get("chunks") or []):
                cid = ck.get("chunk_id", "")
                if cid:
                    ch_by_chunk[cid] = ck.get("highlights") or []

    # Build evidence lists per reference_id
    by_ref: dict[str, list[dict]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        rid = chunk.get("reference_id")
        if not rid:
            continue
        text = chunk.get("content") or chunk.get("text") or ""
        if not text.strip():
            continue  # skip empty chunks

        cid = chunk.get("chunk_id", "")
        highlights = ch_by_chunk.get(cid, [])

        # Collect unique (page, bbox) pairs from highlights
        seen: set[tuple] = set()
        pages_bboxes: list[dict] = []
        for hl in highlights:
            page = hl.get("page", 1)
            bbox = hl.get("bbox") or {}
            key = (page, json.dumps(bbox, sort_keys=True))
            if key not in seen:
                seen.add(key)
                pages_bboxes.append({"page": page, "bbox": bbox})

        evidence = {
            "chunk_id": cid,
            "text": text,
            "pages": pages_bboxes,
        }
        by_ref.setdefault(str(rid), []).append(evidence)

    # Build final map keyed on reference_id
    ref_map: dict[str, str] = {}
    for ref in references:
        if isinstance(ref, dict):
            rid = ref.get("reference_id", "")
            fp = ref.get("file_path", "")
            if rid:
                ref_map[str(rid)] = fp

    citations: dict[str, dict] = {}
    for rid, evs in by_ref.items():
        fp = ref_map.get(rid, "")
        citations[rid] = {
            "reference_id": rid,
            "file_path": fp,
            "evidence": evs,
        }

    return citations


def parse_cited_ids(text: str) -> set[str]:
    """Extract chunk IDs from legacy and block-level inline citations.

    Block citations use ``[^Index:ChunkID§SectionID¶BlockID]``.  The numeric
    index is display-only; filtering must use ``ChunkID`` because that is the
    identifier stored on retrieved chunks and their highlight records.
    """
    cited: set[str] = set()
    for match in re.finditer(r'\[\^([^\]]+)\]', text):
        payload = match.group(1)
        selector = payload.split(":", 1)[1] if ":" in payload else payload
        chunk_id = selector.split("§", 1)[0].strip()
        if chunk_id:
            cited.add(chunk_id)
    return cited


def parse_citation_selectors(text: str) -> list[dict[str, str]]:
    """Parse block-level inline citations in their answer order.

    The returned selector retains the complete citation marker, making every
    emitted visual highlight explicitly traceable to the claim that cited it.
    Legacy ``[^ChunkID]`` citations intentionally have an empty ``block_id``.
    """
    selectors: list[dict[str, str]] = []
    for match in re.finditer(r'\[\^([^\]]+)\]', text):
        payload = match.group(1)
        index, selector = (
            payload.split(":", 1) if ":" in payload else ("", payload)
        )
        chunk_id, section_and_block = (
            selector.split("§", 1) if "§" in selector else (selector, "")
        )
        section_id, block_id = (
            section_and_block.split("¶", 1)
            if "¶" in section_and_block
            else (section_and_block, "")
        )
        chunk_id = chunk_id.strip()
        if not chunk_id:
            continue
        selectors.append(
            {
                "citation_id": match.group(0),
                "index": index.strip(),
                "chunk_id": chunk_id,
                "section_id": section_id.strip(),
                "block_id": block_id.strip(),
            }
        )
    return selectors


def _highlight_block_from_sidecar(file_path: str, block_id: str) -> list[dict]:
    """Resolve one cited parser block directly from its ``blocks.jsonl`` row.

    This is the exact-query fallback for older vector records whose persisted
    ``positions`` lack a ``block_id``. It does not use text matching and never
    returns other blocks on the same page.
    """
    if not file_path or not block_id:
        return []
    try:
        from lightrag.utils_pipeline import parsed_artifact_dir_for

        artifact_dir = parsed_artifact_dir_for(file_path)
        candidates = sorted(artifact_dir.glob("*.blocks.jsonl"))
    except Exception:
        return []
    if not candidates:
        return []

    for block in _load_blocks_jsonl(str(candidates[0])):
        if str(block.get("blockid") or "") != block_id:
            continue
        # A block-level citation selects the complete visual block. Character
        # ranges can further refine this in a future citation grammar.
        start = 0
        end = len(str(block.get("content") or ""))
        highlights = _normalize_positions(
            block.get("positions") or [], block_id=block_id
        )
        for ordinal, highlight in enumerate(highlights):
            highlight["start"] = start
            highlight["end"] = end
            highlight["highlight_id"] = f"hl_{block_id}_{start}_{end}_{ordinal}"
        return highlights
    return []


def filter_to_cited(
    citation_highlights: dict[str, Any] | None,
    citations: dict[str, Any] | None,
    references: list[dict[str, Any]] | None,
    chunks: list[dict[str, Any]] | None,
    cited_ids: set[str],
    citation_selectors: list[dict[str, str]] | None = None,
) -> tuple:
    """Filter citation data to only the referenced content boxes.

    Parameters
    ----------
    citation_highlights:
        The ``citation_highlights`` sidecar (``{version, sources: {file: ...}}``).
    citations:
        The ``citations`` evidence map (``{ref_id: {evidence: [...]}}``).
    references:
        The ``references`` list (``[{reference_id, file_path, ...}]``).
    chunks:
        The ``data["chunks"]`` list (formatted chunk dicts).
    cited_ids:
        Set of reference_id strings actually cited by the LLM.

    Returns
    -------
    tuple
        ``(filtered_highlights, filtered_citations, filtered_references,
        filtered_chunks)``.  ``None`` is returned for inputs that were
        ``None``; empty collections for inputs that produced no matches.
    """
    selectors = citation_selectors or [
        {
            "citation_id": f"[^{chunk_id}]",
            "index": "",
            "chunk_id": chunk_id,
            "section_id": "",
            "block_id": "",
        }
        for chunk_id in cited_ids
    ]

    # --- highlights: keep only the block/range explicitly cited -----------
    # The retrieval reference ID is usually a chunk ID, but the inline
    # citation grammar also permits a durable document ID.  Keep the answer's
    # selector as the public reference ID: it is the only ID a caller can use
    # to relate ``references`` back to the answer without reverse-engineering
    # a retrieved embedding chunk.
    ch_filtered = None
    selector_files: dict[str, list[str]] = {}
    if citation_highlights:
        sources = citation_highlights.get("sources") or {}
        fs: dict[str, Any] = {}
        sidecar_highlight_cache: dict[tuple[str, str], list[dict]] = {}
        for fp, src in sources.items():
            kept: list[dict[str, Any]] = []
            for selector in selectors:
                entry_key = (
                    selector["chunk_id"],
                    selector["section_id"],
                    selector["block_id"],
                )
                if any(entry.get("_key") == entry_key for entry in kept):
                    # A claim may cite a block more than once. One block entry
                    # plus its list of rectangles is sufficient for the viewer.
                    continue
                # Prefer the parser block itself. This handles both new records
                # (which carry ``block_id`` on every stored rectangle) and old
                # records (which only have a chunk-wide position list).
                block_id = selector["block_id"]
                if block_id:
                    cache_key = (fp, block_id)
                    if cache_key not in sidecar_highlight_cache:
                        sidecar_highlight_cache[cache_key] = _highlight_block_from_sidecar(
                            fp, block_id
                        )
                    direct = sidecar_highlight_cache[cache_key]
                    if direct:
                        kept.append(
                            {
                                "_key": entry_key,
                                "chunk_id": selector["chunk_id"],
                                "reference_id": selector["chunk_id"],
                                "citation_id": selector["citation_id"],
                                "section_id": selector["section_id"],
                                "block_id": block_id,
                                # block_id/citation_id live on the containing
                                # entry. Rectangles need only visual geometry.
                                "highlights": _compact_highlights(direct),
                            }
                        )
                        selector_files.setdefault(selector["chunk_id"], []).append(fp)
                        continue
                for chunk in src.get("chunks") or []:
                    candidate_ids = {
                        str(value)
                        for value in (
                            src.get("reference_id"),
                            chunk.get("reference_id"),
                            chunk.get("chunk_id"),
                        )
                        if value
                    }
                    if selector["chunk_id"] not in candidate_ids:
                        continue
                    highlighted = _compact_highlights(
                        highlight
                        for highlight in chunk.get("highlights") or []
                        if not selector["block_id"]
                        or str(highlight.get("block_id")) == selector["block_id"]
                    )
                    if highlighted:
                        kept.append(
                            {
                                "_key": entry_key,
                                "chunk_id": selector["chunk_id"],
                                "reference_id": selector["chunk_id"],
                                "citation_id": selector["citation_id"],
                                "section_id": selector["section_id"],
                                "block_id": selector["block_id"],
                                "highlights": highlighted,
                            }
                        )
                        selector_files.setdefault(selector["chunk_id"], []).append(fp)
            if kept:
                # ``_key`` was only used to deduplicate this response.
                fs[fp] = {
                    "file_path": src.get("file_path", fp),
                    "chunks": [
                        {key: value for key, value in entry.items() if key != "_key"}
                        for entry in kept
                    ],
                }
        ch_filtered = {"version": citation_highlights.get("version", 1), "sources": fs}

    # --- citations map: keep only cited keys -----------------------------
    cit_filtered: dict[str, Any] | None = None
    if citations:
        cit_filtered = {k: v for k, v in citations.items() if k in cited_ids}

    # --- references list: keep only cited entries ------------------------
    refs_filtered: list[dict[str, Any]] | None = None
    if references is not None:
        refs_filtered = []
        seen_reference_ids: set[str] = set()
        for selector in selectors:
            selector_id = selector["chunk_id"]
            if selector_id in seen_reference_ids:
                continue
            paths = selector_files.get(selector_id, [])
            if not paths:
                # Legacy/no-position fallback: retrieve the matching file from
                # the normal reference list when the selector is a chunk ID.
                paths = [
                    str(ref.get("file_path", ""))
                    for ref in references
                    if str(ref.get("reference_id", "")) == selector_id
                ]
            if not paths:
                continue
            seen_reference_ids.add(selector_id)
            original = next(
                (
                    ref
                    for ref in references
                    if str(ref.get("file_path", "")) == paths[0]
                ),
                {},
            )
            refs_filtered.append(
                {
                    **original,
                    "reference_id": selector_id,
                    "file_path": paths[0],
                }
            )

    # --- chunks list: keep only cited entries ----------------------------
    chunks_filtered: list[dict[str, Any]] | None = None
    if chunks:
        chunks_filtered = [
            c for c in chunks
            if str(c.get("reference_id", "")) in cited_ids
        ]

    return ch_filtered, cit_filtered, refs_filtered, chunks_filtered


def _compact_highlights(highlights: Any) -> list[dict[str, Any]]:
    """Return unique PDF rectangles without repeating block/span metadata.

    ``highlight_id`` remains on each rectangle because it is the stable UI
    identity. ``start`` and ``end`` remain too: they are the meaningful
    block-relative character span selected by the citation. ``block_id`` and
    the citation marker live on the surrounding cited-block entry, where they
    apply to every rectangle in the list.
    """
    compact: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for highlight in highlights:
        if not isinstance(highlight, dict):
            continue
        page = highlight.get("page", 1)
        bbox = highlight.get("bbox") or {}
        key = (str(page), json.dumps(bbox, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        highlight_id = str(highlight.get("highlight_id") or "")
        start = highlight.get("start")
        end = highlight.get("end")
        if (
            not highlight_id
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            # Citation-capable records must provide their stable ID and exact
            # block-relative span. Do not synthesize compatibility data.
            continue
        compact.append(
            {
                "page": page,
                "bbox": bbox,
                "start": start,
                "end": end,
                "highlight_id": highlight_id,
            }
        )
    return compact
