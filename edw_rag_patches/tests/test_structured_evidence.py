"""Tests for block-grounded LLM evidence and server-rendered citations."""

from edw_rag_patches.sidecar import (
    build_evidence_targets,
    references_from_evidence,
    render_structured_answer,
)


def test_evidence_targets_expose_canonical_block_text(monkeypatch) -> None:
    monkeypatch.setattr(
        "edw_rag_patches.sidecar._blocks_by_id",
        lambda _file_path: {
            "block-a": {"blockid": "block-a", "content": "Alpha evidence."},
            "block-b": {"blockid": "block-b", "content": "Beta evidence."},
        },
    )

    targets = build_evidence_targets(
        {
            "document_id": "doc-1",
            "file_path": "report.pdf",
            "content": "Ambiguous combined chunk text that is never evidence.",
            "sidecar": {
                "refs": [
                    {"type": "block", "id": "block-a", "start": 0, "end": 15},
                    {"type": "block", "id": "block-b", "start": 0, "end": 14},
                ]
            },
        }
    )

    assert [target["text"] for target in targets] == [
        "Alpha evidence.",
        "Beta evidence.",
    ]
    assert all(target["evidence_id"].startswith("e_") for target in targets)
    assert all("content" not in target for target in targets)


def test_structured_answer_is_validated_and_server_renders_citation() -> None:
    evidence = {
        "e_123": {
            "document_id": "doc-1",
            "section_id": "root",
            "block_id": "block-a",
        }
    }

    rendered = render_structured_answer(
        '{"segments":[{"markdown":"Alpha is supported.","evidence_ids":["e_123"]}]}',
        evidence,
    )

    assert rendered == (
        "Alpha is supported.[^1:doc-1§root¶block-a]",
        [{"markdown": "Alpha is supported.", "evidence_ids": ["e_123"]}],
    )


def test_structured_answer_rejects_unsent_evidence_id() -> None:
    assert render_structured_answer(
        '{"segments":[{"markdown":"Unsupported.","evidence_ids":["e_fake"]}]}',
        {},
    ) is None


def test_structured_no_context_segment_needs_no_evidence() -> None:
    assert render_structured_answer(
        '{"segments":[{"markdown":"I do not have enough information.","evidence_ids":[]}]}',
        {},
    ) == (
        "I do not have enough information.",
        [{"markdown": "I do not have enough information.", "evidence_ids": []}],
    )


def test_references_are_derived_from_selected_blocks() -> None:
    assert references_from_evidence(
        [
            {"chunk_id": "doc-1", "block_id": "block-a"},
            {"chunk_id": "doc-1", "block_id": "block-b"},
        ],
        {
            "e_a": {
                "document_id": "doc-1",
                "block_id": "block-a",
                "file_path": "report.pdf",
            },
            "e_b": {
                "document_id": "doc-1",
                "block_id": "block-b",
                "file_path": "report.pdf",
            },
        },
    ) == [{"reference_id": "doc-1", "file_path": "report.pdf"}]
