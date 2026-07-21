"""Opt-in native Qdrant dense + BM25 hybrid retrieval.

The core Qdrant backend remains unchanged.  When
``EDW_QDRANT_HYBRID_ENABLED=true``, this module augments each Qdrant vector
collection with a named sparse BM25 vector and fuses its lexical ranking with
the existing dense embedding ranking using Qdrant's RRF query API.

No dense vectors are migrated or recomputed.  Sparse vectors are added only to
newly flushed points; an existing populated collection keeps serving dense
results until it is re-ingested by the caller.
"""

from __future__ import annotations

import os
from typing import Any


BM25_VECTOR_NAME = "edw_bm25"
BM25_MODEL = "qdrant/bm25"


def enabled() -> bool:
    return os.getenv("EDW_QDRANT_HYBRID_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def apply_qdrant_hybrid_patch(
    storage_cls: type, models: Any, logger: Any
) -> dict[str, Any]:
    """Patch a Qdrant storage class and return originals for restoration.

    ``storage_cls`` and ``models`` are injected to keep this module isolated
    from core LightRAG and easy to test with mocks.
    """
    originals = {
        "initialize": storage_cls.initialize,
        "upsert": storage_cls.upsert,
        "delete": storage_cls.delete,
        "flush": storage_cls._flush_pending_vector_ops,
        "query": storage_cls.query,
    }

    async def initialize(self) -> None:
        await originals["initialize"](self)
        if enabled():
            _ensure_sparse_vector(self._client, self.final_namespace, models)

    async def flush(self) -> None:
        # The side buffer is recorded by ``upsert`` rather than copied from
        # core's protected buffer here. That prevents a concurrent enqueue
        # from being dense-indexed but missing its sparse representation.
        sparse_docs = list(getattr(self, "_edw_bm25_pending", {}).items())
        await originals["flush"](self)
        if enabled() and sparse_docs:
            _update_sparse_vectors(self, sparse_docs, models)
            for doc_id, _content in sparse_docs:
                self._edw_bm25_pending.pop(doc_id, None)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        await originals["upsert"](self, data)
        if enabled():
            pending = getattr(self, "_edw_bm25_pending", None)
            if pending is None:
                pending = self._edw_bm25_pending = {}
            for doc_id, value in data.items():
                content = value.get("content")
                if isinstance(content, str) and content:
                    pending[doc_id] = content

    async def delete(self, ids: list[str]) -> None:
        await originals["delete"](self, ids)
        pending = getattr(self, "_edw_bm25_pending", None)
        if pending:
            for doc_id in ids:
                pending.pop(doc_id, None)

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] | None = None
    ) -> list[dict[str, Any]]:
        if not enabled():
            return await originals["query"](self, query, top_k, query_embedding)

        if query_embedding is not None:
            embedding = query_embedding
        else:
            embedding_result = await self.embedding_func(
                [query], context="query", _priority=5
            )
            embedding = embedding_result[0]

        prefetch_limit = max(top_k, top_k * _prefetch_multiplier())
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="workspace_id",
                    match=models.MatchValue(value=self.effective_workspace),
                )
            ]
        )
        dense_prefetch = models.Prefetch(
            query=embedding,
            limit=prefetch_limit,
            score_threshold=self.cosine_better_than_threshold,
        )
        sparse_prefetch = models.Prefetch(
            query=_document(query, models),
            using=_vector_name(),
            limit=prefetch_limit,
        )
        try:
            response = self._client.query_points(
                collection_name=self.final_namespace,
                prefetch=[dense_prefetch, sparse_prefetch],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
                query_filter=query_filter,
            )
        except Exception as exc:
            raise RuntimeError(
                "Qdrant hybrid search failed. Ensure the server supports native "
                "inference and the collection has the configured sparse vector."
            ) from exc

        return [
            {
                **point.payload,
                "distance": point.score,
                "created_at": point.payload.get("created_at"),
            }
            for point in response.points
        ]

    storage_cls.initialize = initialize
    storage_cls.upsert = upsert
    storage_cls.delete = delete
    storage_cls._flush_pending_vector_ops = flush
    storage_cls.query = query
    logger.info("[edw-rag] installed opt-in Qdrant dense + BM25 hybrid patch")
    return originals


def _ensure_sparse_vector(client: Any, collection_name: str, models: Any) -> None:
    """Add the sparse schema once; this is a no-data schema operation."""
    vector_name = _vector_name()
    info = client.get_collection(collection_name)
    sparse_vectors = getattr(info.config.params, "sparse_vectors", None) or {}
    if vector_name in sparse_vectors:
        return
    try:
        client.create_vector_name(
            collection_name=collection_name,
            vector_name=vector_name,
            vector_name_config=models.SparseVectorNameConfig(
                sparse=models.SparseVectorConfig(modifier=models.Modifier.IDF)
            ),
        )
    except AttributeError as exc:
        raise RuntimeError(
            "EDW Qdrant hybrid search requires Qdrant server and qdrant-client "
            "version 1.18+ to add the sparse vector schema."
        ) from exc


def _update_sparse_vectors(self: Any, docs: list[tuple[str, str]], models: Any) -> None:
    points = [
        models.PointVectors(
            id=_qdrant_point_id(self, doc_id),
            vector={_vector_name(): _document(content, models)},
        )
        for doc_id, content in docs
    ]
    self._client.update_vectors(
        collection_name=self.final_namespace,
        points=points,
        wait=True,
    )


def _qdrant_point_id(storage: Any, doc_id: str) -> str:
    # Importing this helper from the core module is read-only usage; the patch
    # never changes LightRAG's Qdrant implementation.
    from lightrag.kg.qdrant_impl import compute_mdhash_id_for_qdrant

    return compute_mdhash_id_for_qdrant(doc_id, prefix=storage.effective_workspace)


def _document(text: str, models: Any) -> Any:
    options: dict[str, Any] = {
        "language": os.getenv("EDW_QDRANT_BM25_LANGUAGE", "none"),
        "tokenizer": os.getenv("EDW_QDRANT_BM25_TOKENIZER", "multilingual"),
        "ascii_folding": _env_bool("EDW_QDRANT_BM25_ASCII_FOLDING", True),
    }
    return models.Document(text=text, model=BM25_MODEL, options=options)


def _vector_name() -> str:
    return os.getenv("EDW_QDRANT_BM25_VECTOR_NAME", BM25_VECTOR_NAME).strip() or BM25_VECTOR_NAME


def _prefetch_multiplier() -> int:
    try:
        return max(1, int(os.getenv("EDW_QDRANT_HYBRID_PREFETCH_MULTIPLIER", "3")))
    except ValueError:
        return 3


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
