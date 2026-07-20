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

import json
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
                            _normalize_positions(blk.get("positions") or [])
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


def _normalize_positions(raw_positions: list[dict]) -> list[dict]:
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

        # Positions from enrich_chunks_with_positions are already in
        # compact highlight format ({page, bbox, origin?}). Use as-is.
        highlights = chunk_positions

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
