"""Test-only fixtures owned by the EDW-RAG patch suite."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
from lightrag.pipeline import _BatchRunContext
from lightrag.parser.registry import parser_specs_snapshot
from lightrag.utils import EmbeddingFunc, Tokenizer


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 8)


async def _noop_llm(prompt, **kwargs):  # pragma: no cover - never invoked
    return ""


def build_rag(tmp_path: Path) -> LightRAG:
    role_configs = {spec.name: RoleLLMConfig() for spec in ROLES}
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"edw-patch-{tmp_path.name}",
        llm_model_func=_noop_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=1024,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        role_llm_configs=role_configs,
    )


async def make_ctx(rag: LightRAG) -> _BatchRunContext:
    pipeline_status = await get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status.clear()
    pipeline_status.update(
        {
            "busy": True,
            "history_messages": [],
            "latest_message": "",
            "cancellation_requested": False,
        }
    )
    return _BatchRunContext(
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        semaphore=asyncio.Semaphore(2),
        total_files=1,
        parse_queues={
            "native": asyncio.Queue(),
            "mineru": asyncio.Queue(),
            "docling": asyncio.Queue(),
        },
        parser_specs=parser_specs_snapshot(),
        q_analyze=asyncio.Queue(),
        q_process=asyncio.Queue(),
    )


def make_status_doc(doc_id: str, *, content_hash: str) -> DocProcessingStatus:
    now = datetime.now(timezone.utc).isoformat()
    return DocProcessingStatus(
        content_summary="stale-summary",
        content_length=999,
        file_path=f"{doc_id}.txt",
        status=DocStatus.PENDING,
        created_at=now,
        updated_at=now,
        track_id=None,
        content_hash=content_hash,
    )


async def seed_doc_status(
    rag: LightRAG, doc_id: str, *, process_options: str = ""
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    await rag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.PENDING.value,
                "content_summary": "stale-summary",
                "content_length": 999,
                "file_path": f"{doc_id}.txt",
                "created_at": now,
                "updated_at": now,
                "track_id": "t",
                "metadata": {"process_options": process_options},
            }
        }
    )
