"""Regression coverage for pipeline chunk-position enrichment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from lightrag.constants import FULL_DOCS_FORMAT_RAW

import lightrag.pipeline as pipeline_module
import edw_rag_patches as patches
from tests.pipeline.test_pipeline_content_reread import (
    _build_rag,
    _make_ctx,
    _make_status_doc,
    _seed_doc_status,
)


def test_pipeline_chunk_builder_receives_blocks_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline's imported builder alias must use the active patch.

    ``lightrag.pipeline`` imports the builder directly, so patching only
    ``lightrag.utils_pipeline`` leaves the pipeline on the original function.
    The explicit path also makes enrichment independent of task context.
    """
    blocks_path = tmp_path / "doc.blocks.jsonl"
    blocks_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "meta", "format": "lightrag"}),
                json.dumps(
                    {
                        "type": "content",
                        "blockid": "b1",
                        "content": "Alpha body.",
                        "positions": [
                            {
                                "type": "bbox",
                                "anchor": "2",
                                "range": [10, 20, 30, 40],
                            }
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    original_pipeline_builder = (
        pipeline_module.build_chunks_dict_from_chunking_result
    )
    monkeypatch.setattr("sys.argv", ["pytest"])
    patches.apply_edw_rag_patches()
    try:
        assert (
            pipeline_module.build_chunks_dict_from_chunking_result
            is patches._build_chunks_dict_patched
        )
        chunks = pipeline_module.build_chunks_dict_from_chunking_result(
            [{"content": "Alpha body.", "chunk_order_index": 0}],
            doc_id="doc-1",
            file_path="doc.pdf",
            blocks_path=str(blocks_path),
        )
    finally:
        patches.revert_edw_rag_patches()

    assert pipeline_module.build_chunks_dict_from_chunking_result is original_pipeline_builder
    assert chunks["doc-1-chunk-000"]["positions"] == [
        {
            "page": 2,
            "bbox": {"l": 10.0, "t": 20.0, "r": 30.0, "b": 40.0},
        }
    ]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_pipeline_process_persists_bbox_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real process stage, not only the patched helper."""
    blocks_path = tmp_path / "doc.blocks.jsonl"
    blocks_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "meta", "format": "lightrag"}),
                json.dumps(
                    {
                        "type": "content",
                        "blockid": "b1",
                        "content": "Alpha body.",
                        "positions": [
                            {
                                "type": "bbox",
                                "anchor": "2",
                                "range": [10, 20, 30, 40],
                            }
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["pytest"])
    patches.apply_edw_rag_patches()
    rag = _build_rag(tmp_path / "rag")
    await rag.initialize_storages()
    try:
        doc_id = "doc-pipeline-bbox"
        await rag.full_docs.upsert(
            {
                doc_id: {
                    "content": "Alpha body.",
                    "file_path": "doc.pdf",
                    "parse_format": FULL_DOCS_FORMAT_RAW,
                    "process_options": "!",
                }
            }
        )
        await _seed_doc_status(rag, doc_id, process_options="!")
        ctx = await _make_ctx(rag)
        await rag.process_single_document(
            doc_id=doc_id,
            status_doc=_make_status_doc(doc_id, content_hash="hash-bbox"),
            parsed_data={
                "doc_id": doc_id,
                "file_path": "doc.pdf",
                "blocks_path": str(blocks_path),
            },
            ctx=ctx,
        )

        row = await rag.doc_status.get_by_id(doc_id)
        assert row is not None
        chunks = await rag.text_chunks.get_by_ids(row["chunks_list"])
        assert chunks
        assert chunks[0]["positions"] == [
            {
                "page": 2,
                "bbox": {"l": 10.0, "t": 20.0, "r": 30.0, "b": 40.0},
            }
        ]
    finally:
        await rag.finalize_storages()
        patches.revert_edw_rag_patches()
