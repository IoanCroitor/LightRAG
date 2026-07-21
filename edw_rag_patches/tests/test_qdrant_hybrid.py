"""Unit coverage for the patch-local Qdrant hybrid adapter."""

from types import SimpleNamespace

from edw_rag_patches import qdrant_hybrid


class _Document:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Prefetch:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FusionQuery:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _PointVectors:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _SparseVectorConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _SparseVectorNameConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Filter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FieldCondition:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _MatchValue:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Models:
    Document = _Document
    Prefetch = _Prefetch
    FusionQuery = _FusionQuery
    PointVectors = _PointVectors
    SparseVectorConfig = _SparseVectorConfig
    SparseVectorNameConfig = _SparseVectorNameConfig
    Filter = _Filter
    FieldCondition = _FieldCondition
    MatchValue = _MatchValue
    Modifier = SimpleNamespace(IDF="idf")
    Fusion = SimpleNamespace(RRF="rrf")


class _Client:
    def __init__(self):
        self.create_vector_name_calls = []
        self.update_vectors_calls = []
        self.query_points_calls = []
        self.collection_info = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(sparse_vectors={}))
        )

    def get_collection(self, _collection_name):
        return self.collection_info

    def create_vector_name(self, **kwargs):
        self.create_vector_name_calls.append(kwargs)

    def update_vectors(self, **kwargs):
        self.update_vectors_calls.append(kwargs)

    def query_points(self, **kwargs):
        self.query_points_calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-1",
                    payload={"id": "chunk-1", "created_at": 1},
                    score=0.42,
                )
            ]
        )


class _Storage:
    def __init__(self):
        self._client = _Client()
        self.final_namespace = "chunks"
        self.effective_workspace = "workspace"
        self.cosine_better_than_threshold = 0.2
        self._pending_vector_docs = {}

        async def embedding(texts, **_kwargs):
            assert texts == ["ABC-123"]
            return [[0.1, 0.2]]

        self.embedding_func = embedding

    async def initialize(self):
        return None

    async def _flush_pending_vector_ops(self):
        self._pending_vector_docs.clear()

    async def upsert(self, data):
        self._pending_vector_docs.update(
            {
                doc_id: SimpleNamespace(content=value["content"])
                for doc_id, value in data.items()
            }
        )

    async def delete(self, ids):
        for doc_id in ids:
            self._pending_vector_docs.pop(doc_id, None)

    async def get_vectors_by_ids(self, _ids):
        return {
            "chunk-1": {
                "": [0.1, 0.2],
                "edw_bm25": {"indices": [1], "values": [0.7]},
            }
        }

    async def query(self, *_args, **_kwargs):
        return [{"legacy": True}]


def _restore(storage_cls, originals) -> None:
    storage_cls.initialize = originals["initialize"]
    storage_cls.upsert = originals["upsert"]
    storage_cls.delete = originals["delete"]
    storage_cls._flush_pending_vector_ops = originals["flush"]
    storage_cls.get_vectors_by_ids = originals["get_vectors_by_ids"]
    storage_cls.query = originals["query"]


async def test_native_bm25_is_added_without_reembedding_dense_vectors(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EDW_QDRANT_HYBRID_ENABLED", "true")
    monkeypatch.setattr(qdrant_hybrid, "_qdrant_point_id", lambda _s, _id: "uuid-1")
    originals = qdrant_hybrid.apply_qdrant_hybrid_patch(
        _Storage, _Models, SimpleNamespace(info=lambda _m: None)
    )
    storage = _Storage()

    await storage.initialize()
    await storage.upsert({"chunk-1": {"content": "Exact project identifier ABC-123"}})
    await storage._flush_pending_vector_ops()
    result = await storage.query("ABC-123", top_k=2)
    vectors = await storage.get_vectors_by_ids(["chunk-1"])

    assert storage._client.create_vector_name_calls[0]["vector_name"] == "edw_bm25"
    point = storage._client.update_vectors_calls[0]["points"][0]
    assert point.kwargs["vector"]["edw_bm25"].kwargs["text"] == "Exact project identifier ABC-123"
    query = storage._client.query_points_calls[0]
    assert len(query["prefetch"]) == 2
    assert query["query"].kwargs["fusion"] == "rrf"
    assert result == [{"id": "chunk-1", "created_at": 1, "distance": 0.42}]
    assert vectors == {"chunk-1": [0.1, 0.2]}
    _restore(_Storage, originals)


async def test_debug_mode_logs_dense_bm25_and_fused_rankings(monkeypatch) -> None:
    monkeypatch.setenv("EDW_QDRANT_HYBRID_ENABLED", "true")
    monkeypatch.setenv("EDW_QDRANT_HYBRID_DEBUG", "true")
    messages: list[str] = []
    logger = SimpleNamespace(
        info=lambda _m: None,
        debug=lambda message, *args: messages.append(message % args),
        exception=lambda message, *args: messages.append(message % args),
    )
    originals = qdrant_hybrid.apply_qdrant_hybrid_patch(_Storage, _Models, logger)
    storage = _Storage()

    await storage.query("ABC-123", top_k=2)

    # One fused RRF request plus dense and BM25 review requests in debug mode.
    assert len(storage._client.query_points_calls) == 3
    assert any("branch=fused_rrf" in message for message in messages)
    assert any("branch=dense" in message for message in messages)
    assert any("branch=bm25" in message for message in messages)
    assert all("ABC-123" not in message for message in messages)
    _restore(_Storage, originals)
