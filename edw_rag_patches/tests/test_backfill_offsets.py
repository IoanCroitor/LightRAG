"""Regression coverage for EDW's runtime block-span backfill patch."""

import json

import lightrag.sidecar.backfill as upstream_backfill

from edw_rag_patches import _originals, _patch_sidecar_backfill


def test_runtime_backfill_keeps_block_relative_offsets(tmp_path) -> None:
    """The upstream file remains unchanged; the patch supplies exact spans."""
    blocks_path = tmp_path / "report.blocks.jsonl"
    blocks_path.write_text(
        "\n".join(
            json.dumps({"type": "content", "blockid": block_id, "content": text})
            for block_id, text in [("b1", "Alpha paragraph."), ("b2", "Beta paragraph.")]
        )
        + "\n",
        encoding="utf-8",
    )
    separator = upstream_backfill._BLOCK_SEPARATOR
    merged = separator.join(["Alpha paragraph.", "Beta paragraph."])
    chunks = [
        {
            "content": merged,
            "chunk_order_index": 0,
            "_source_span": {"start": 0, "end": len(merged)},
        }
    ]

    original = upstream_backfill.backfill_chunk_sidecars
    _patch_sidecar_backfill(upstream_backfill)
    try:
        upstream_backfill.backfill_chunk_sidecars(chunks, str(blocks_path))
    finally:
        upstream_backfill.backfill_chunk_sidecars = original
        _originals.pop("backfill_chunk_sidecars", None)

    assert chunks[0]["sidecar"]["refs"] == [
        {"type": "block", "id": "b1", "start": 0, "end": len("Alpha paragraph.")},
        {"type": "block", "id": "b2", "start": 0, "end": len("Beta paragraph.")},
    ]
