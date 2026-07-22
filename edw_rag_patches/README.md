# EDW-RAG Patches -- Chunk-Level Citation Highlights for LightRAG

We monkey-patch because we need the patches to apply transparently - the server, the CLI, the WebUI, and the SDK all create LightRAG() instances, and we want our logic in every path without rewriting any of those call sites.

---

## The Problem

LightRAG returns **document-level** citations only -- a list of file paths with reference IDs:

```json
"references": [{"reference_id": "1", "file_path": "/doc/ai.pdf"}]
```

That tells you *which document* the answer came from, but not *where* in that
document. For a PDF with hundreds of pages, that's not actionable.

## The Solution

These patches store **bounding-box positions** (page number + `[l, t, r, b]`
coordinates) for every chunk at ingestion time, then expose them as a
structured **citation_highlights** sidecar in query responses:

```json
{
  "version": 1,
  "sources": {
    "/doc/ai.pdf": {
      "file_path": "/doc/ai.pdf",
      "reference_id": "1",
      "chunks": [
        {
          "chunk_id": "chunk-abc123",
          "reference_id": "1",
          "highlights": [
            {"page": 1, "bbox": {"l": 72, "t": 200, "r": 523, "b": 215}},
            {"page": 1, "bbox": {"l": 72, "t": 215, "r": 500, "b": 230}}
          ]
        }
      ]
    }
  }
}
```

A custom PDF viewer receives this + the original PDF, and highlights every
`bbox` on its `page` -- independent of how the LLM rendered the answer into
text.

---

## Data Flow

```mermaid
flowchart LR
    PDF -->|Docling| PAR[.parsed/blocks.jsonl]
    PAR -->|enrich_chunks_with_positions| STORE[chunks_vdb + text_chunks]
    STORE -->|query| CTX[_get_vector_context]
    CTX -->|carries positions| MERGE[_merge_all_chunks]
    MERGE -->|preserves positions| FMT[convert_to_user_format]
    FMT -->|builds| SIDECAR[citation_highlights]
    SIDECAR -->|/query/data| API[JSON Response]
```

| Stage | What happens | Where |
|-------|-------------|-------|
| **Parse** | Docling produces `.parsed/stem.blocks.jsonl` with per-item `prov[]` arrays (page, bbox, charspan, origin) | `lightrag/parser/external/docling/` |
| **Chunk** | P-chunker produces `sidecar.refs` block IDs; F/R/V produce `_source_span` char offsets | `lightrag/chunker/` |
| **Enrich** | `enrich_chunks_with_positions()` maps each chunk back to block positions from `blocks.jsonl` | `edw_rag_patches/sidecar.py` |
| **Store** | Positions are stored in VDB meta alongside chunk content -- no second lookup | `edw_rag_patches/__init__.py` |
| **Retrieve** | `get_vector_context()` extracts `positions` from VDB results | `edw_rag_patches/query_chain.py` |
| **Merge** | `merge_all_chunks()` carries `positions` through round-robin dedup | `edw_rag_patches/query_chain.py` |
| **Format** | `convert_to_user_format()` builds the `citation_highlights` sidecar | `edw_rag_patches/query_chain.py` |
| **API** | `/query/data` returns it inside `data["citation_highlights"]` | Built-in |

---

## Architecture

### Patch strategy: monkey-patching, not forking

```
edw_rag_patches/
├── __init__.py        # apply/revert logic + all 8 monkey-patches
├── sidecar.py         # chunk→bbox mapping + sidecar builder
├── query_chain.py     # query pipeline patches
├── cli.py             # entrypoint: edw-rag-server
└── gunicorn.py        # entrypoint: edw-rag-gunicorn
```

## Reliability TODO

Before treating these patches as production-ready, address the following:

- [ ] Preserve the exact upstream `LightRAG.aquery_llm` signature in its wrapper, including positional `system_prompt`, `progress_callback`, and the default `QueryParam` behavior.
- [x] Patch application is idempotent. A second `apply_edw_rag_patches()` call is a no-op and cannot save a patched function as its own original.
- [ ] Complete `revert_edw_rag_patches()`: restore patched prompts and the `query_routes.QueryParam` alias, as well as every other process-global mutation.
- [ ] Replace the `_source_span` fallback that returns every document block with an offset-aware block intersection. If no reliable mapping is available, fall back to text overlap rather than emitting unrelated page overlays.
- [ ] Honor `include_citation_highlights`. Generate and expose sidecars only when callers opt in, including a well-defined streaming response format.
- [ ] Patch API response models before routes are built, or rebuild FastAPI response fields after patching existing routes so `citation_highlights` is not filtered during serialization.
- [ ] Add regression tests for the wrapper signature, repeated apply/revert cycles, opt-in behavior, route serialization, and source-span position selection.

`revert_edw_rag_patches()` currently restores only a subset of mutations; do not rely on it for a clean process reset until the TODOs above are complete. On upgrade:

```bash
# 1. Update LightRAG
uv sync --extra api

# 2. Run the isolated EDW patch regressions
./edw_rag_patches/ci/run_edw_pipeline.sh patch-tests

# 3. If a test fails, fix the signature drift in the corresponding
#    patch function -- no LightRAG source files were ever touched.
```

### Patch propagation

The pipeline passes the parsed Docling ``blocks_path`` directly into
``build_chunks_dict_from_chunking_result``.  This avoids relying on a
``contextvars.ContextVar`` surviving task creation and await boundaries.

| Context var | What it carries | Set by | Consumed by |
|------------|----------------|--------|-------------|
| `_citation_cv` | Citation highlights dict | Patched `aquery_llm` | Route handler |

---

## Quick Start

### 1. Install

```bash
cd ~/Documents/edw_rag/LightRAG

# Dependencies (LightRAG + API)
uv sync --extra api
```

### 2. Configure

Create `.env` (example provided in repo):

```ini
LLM_BINDING=openai
LLM_BINDING_HOST=http://fdz2.edw.ro/v1
LLM_BINDING_API_KEY=your-key-here
LLM_BINDING_MODEL=qwen3.6-35b-a3b-fast

EMBEDDING_BINDING=openai
EMBEDDING_BINDING_HOST=http://fdz2.edw.ro/v1
EMBEDDING_BINDING_API_KEY=your-key-here
EMBEDDING_BINDING_MODEL=qwen3-embedding-0.6b
EMBEDDING_DIM=1024

RERANK_BINDING=openai
RERANK_BINDING_HOST=http://fdz2.edw.ro/v1
RERANK_BINDING_API_KEY=your-key-here
RERANK_BINDING_MODEL=qwen3-reranker-0.6b
```

### Native Qdrant hybrid search diagnostics

When native Qdrant hybrid search is enabled, set this only while diagnosing a
query:

```ini
EDW_QDRANT_HYBRID_ENABLED=true
EDW_QDRANT_HYBRID_DEBUG=true
# Opt in only when query text is safe to write to application logs.
EDW_QDRANT_HYBRID_LOG_QUERY_TEXT=true
```

At the application's `DEBUG` log level, every hybrid query logs its collection,
workspace, limits, and the ranked `dense`, `bm25`, and fused-RRF result IDs and
scores. Query text is excluded unless `EDW_QDRANT_HYBRID_LOG_QUERY_TEXT=true`;
chunk text is always excluded. Debug mode makes two
additional read-only Qdrant queries per request to expose the independent
dense and BM25 rankings; disable it after review.

### 3. Run

```bash
# Single process (dev or small workloads)
uv run python -m edw_rag_patches.cli --host 127.0.0.1 --port 9621

# Multi-worker (production)
uv run python -m edw_rag_patches.gunicorn --workers 4 --port 9621

# Or just test the pipeline with sample documents
uv run python examples/edw_rag_demo.py
```

### 4. Query

```bash
curl -X POST http://127.0.0.1:9621/query/data \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "mode": "mix"}'
```

The response includes `data.citation_highlights` with per-chunk highlights.

---

## API

### Patch management

```python
from edw_rag_patches import (
    apply_edw_rag_patches,      # install all patches
    revert_edw_rag_patches,     # restore all originals
    patch_app_routes,           # patch FastAPI app routes for /query
)
```

### Extensions to LightRAG models

| Model | New field | Description |
|-------|-----------|-------------|
| `QueryParam` | `include_citation_highlights: bool` | Request flag (default `False`) |
| `QueryRequest` | `include_citation_highlights: bool` | API request field (default `False`) |
| `QueryResponse` | `citation_highlights: dict \| None` | API response field |

### Sidecar JSON schema

```json
{
  "version": 1,
  "sources": {
    "<file_path>": {
      "file_path": "<document path>",
      "reference_id": "<1-based reference number>",
      "chunks": [
        {
          "chunk_id": "<chunk identifier>",
          "reference_id": "<same reference number>",
          "highlights": [
            {
              "page": <integer page number>,
              "bbox": {
                "l": <float left>,
                "t": <float top>,
                "r": <float right>,
                "b": <float bottom>
              },
              "origin": "LEFTTOP"   // optional, default coordinate system
            }
          ]
        }
      ]
    }
  }
}
```

---

## Testing

```bash
# Quick patch integrity check (< 1s)
uv run python -c "
from edw_rag_patches import apply_edw_rag_patches
apply_edw_rag_patches()
import lightrag.utils_pipeline as lup
assert lup.build_chunks_dict_from_chunking_result.__module__ == 'edw_rag_patches'
print('Patches OK')
"

# Full EDW patch regression suite
./edw_rag_patches/ci/run_edw_pipeline.sh patch-tests

# Functional test: sidecar JSON generation
uv run python -c "
import tempfile, json, os
from edw_rag_patches.sidecar import enrich_chunks_with_positions, build_citation_highlights
blocks = os.path.join(tempfile.mkdtemp(), 'blocks.jsonl')
with open(blocks, 'w') as f:
    f.write(json.dumps({'type': 'content', 'blockid': 'b1', 'content': 'test text',
        'positions': [{'type': 'bbox', 'anchor': '1', 'range': [0,0,100,50]}]}) + '\n')
chunks = {'ch-000': {'content': 'test text', 'file_path': '/doc.pdf', 'chunk_order_index': 0}}
enrich_chunks_with_positions(chunks, [{'content': 'test text', 'chunk_order_index': 0,
    'sidecar': {'type': 'block', 'id': 'b1', 'refs': [{'type': 'block', 'id': 'b1'}]}}], blocks)
assert 'positions' in chunks['ch-000']
print('Functional test OK')
"
```

---

## CI/CD (GitLab)

`.gitlab-ci.yml` verifies monkey-patch targets in the process that installs
them. EDW patch tests live exclusively in `edw_rag_patches/tests/` and begin
unpatched so they can test apply/revert cycles themselves.

| Stage | Job | When it runs |
|-------|-----|--------------|
| `lint` | `ruff-lint` | Every branch and merge-request pipeline |
| `test` | `patch-tests` | Focused EDW patch regressions |
| `integration` | `parsebench` | Only with `RUN_PARSEBENCH=true` |

Run the same jobs locally after `uv sync --extra test`:

```bash
./edw_rag_patches/ci/run_edw_pipeline.sh lint
./edw_rag_patches/ci/run_edw_pipeline.sh patch-tests
```

### Optional ParseBench integration

`parsebench` is intentionally disabled by default. When enabled, GitLab builds
an image containing the current EDW patch overlay, starts it as the disposable
`lightrag` service, and targets `http://lightrag:9621` over the job network.
It clears and re-ingests that service; it does not contact a production server.
Configure the following masked GitLab variables, then start a pipeline with
`RUN_PARSEBENCH=true`:

| Variable | Purpose |
|----------|---------|
| `LIGHTRAG_LLM_BASE_URL`, `LIGHTRAG_LLM_API_KEY`, `LIGHTRAG_LLM_MODEL` | LLM configuration for the in-CI LightRAG service |
| `LIGHTRAG_EMBEDDING_BASE_URL`, `LIGHTRAG_EMBEDDING_API_KEY`, `LIGHTRAG_EMBEDDING_MODEL`, `LIGHTRAG_EMBEDDING_DIM` | Embedding configuration for the in-CI LightRAG service |
| `LIGHTRAG_API_KEY` | Optional API key when authentication is enabled on the service |
| `LLM_API_KEY` or `OPENROUTER_API_KEY` | Ground-truth answer judge credential |
| `LLM_BASE_URL`, `LLM_MODEL`, `JUDGE_MODEL` | Optional judge endpoint/model overrides |

The job first clears the target server's in-memory and persistent LLM cache,
then uploads the versioned corpus, evaluates against the versioned ground
truth, checks citation highlights, and publishes `artifacts/parsebench/` even
when evaluation fails. To run it locally against a server already started with
the EDW patches:

```bash
export LIGHTRAG_BASE_URL=http://127.0.0.1:9621
export LLM_API_KEY=...  # or OPENROUTER_API_KEY
./edw_rag_patches/ci/run_edw_pipeline.sh parsebench
```

---

## How it Works: Position Matching

### P-chunker (paragraph_semantic)

The P-chunker already reads `blocks.jsonl` and produces `sidecar.refs` --
a list of block IDs that were merged into each chunk.  Our enrichment is
a direct ID lookup:

```
chunk.sidecar.refs  →  [block-001, block-002]
                            ↓              ↓
blocks.jsonl:  block-001.{positions}   block-002.{positions}
                            ↓              ↓
chunk.positions = [page=1, bbox={...}] + [page=1, bbox={...}]
```

### F/R/V chunkers (fixed_token / recursive_character / semantic_vector)

These chunkers don't know about blocks.  We fall back to **text-overlap
matching**: for each chunk, find blocks whose content shares ≥20 characters
with the chunk, and take their positions.  Reliable for verbatim chunking
because chunk text is a substring of the original document.

```
chunk "Artificial Intelligence is..."  →  find blocks with matching text
                                            ↓
blocks.jsonl:  block-003 has "Artificial Intelligence is..."
                                            ↓
chunk.positions = [page=1, bbox={...}]
```

---

## Upgrade / Maintenance

```bash
# Update LightRAG
uv sync --extra api

# Run patch integrity check
uv run python -c "
from edw_rag_patches import apply_edw_rag_patches
apply_edw_rag_patches()
import lightrag.utils_pipeline as lup
import lightrag.operate as lo
import lightrag.utils as lu
import lightrag.base as lb

assert lup.build_chunks_dict_from_chunking_result.__module__ == 'edw_rag_patches'
assert lo._get_vector_context.__name__ == 'get_vector_context'
assert lu.convert_to_user_format.__module__ == 'edw_rag_patches.query_chain'
assert hasattr(lb.QueryParam, 'include_citation_highlights')
print('All patches intact after upgrade')
"

# If a test fails, the fix is in one file:
#   edw_rag_patches/__init__.py      -- adjust import paths or signatures
#   edw_rag_patches/query_chain.py   -- adjust function signatures
#   edw_rag_patches/sidecar.py        -- adjust position format

# LightRAG source files are never modified.
```

---

## Instructions for Cache Deletion

### 1. Via REST API (Recommended if running the server)

Call the `POST /documents/clear_cache` endpoint to clear the in-memory and persistent LLM response cache:

```bash
curl -X 'POST' \
  'http://localhost:9621/documents/clear_cache' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{}'
```

If your server requires an API key, add `-H "X-API-Key: YOUR_API_KEY"`.

### 2. File System Deletion (For File-based KV Storage)

If using default JSON storage (`working_dir="./rag_storage"`), clear or delete the cache file:

```bash
# Overwrite with empty JSON object
echo "{}" > ./rag_storage/kv_store_llm_response_cache.json

# Or remove the cache file directly
rm -f ./rag_storage/kv_store_llm_response_cache.json
```

If running via Docker Compose:

```bash
docker exec -it <container_name> rm -f ./rag_storage/kv_store_llm_response_cache.json
```

### 3. Via Python SDK

In your Python script with an initialized `LightRAG` instance:

```python
await rag.aclear_cache()
```

### 4. Database Storage Backends

If `KV_STORAGE` is set to an external database backend:

- **PostgreSQL (`PGKVStorage`)**:
  ```sql
  TRUNCATE TABLE LIGHTRAG_LLM_CACHE;
  ```
- **Redis (`RedisKVStorage`)**:
  ```bash
  redis-cli --eval "return redis.call('del', unpack(redis.call('keys', '*llm_response_cache*')))"
  ```

---

## File Reference

| File | Purpose |
|------|---------|
| `edw_rag_patches/__init__.py` | Patch manager: `apply`, `revert`, `patch_app_routes`. Contains all 8 monkey-patches and API model extensions. |
| `edw_rag_patches/sidecar.py` | Core position logic: `enrich_chunks_with_positions()`, `build_citation_highlights()`, `_normalize_positions()`, `_load_blocks_jsonl()`, text-overlap matching. |
| `edw_rag_patches/query_chain.py` | Query pipeline patches: `get_vector_context()`, `merge_all_chunks()`, `convert_to_user_format()`. |
| `edw_rag_patches/cli.py` | Single-process server entrypoint. Usage: `uv run python -m edw_rag_patches.cli` |
| `edw_rag_patches/gunicorn.py` | Multi-worker production entrypoint. Usage: `uv run python -m edw_rag_patches.gunicorn` |
| `edw_rag_patches/ci/` | Isolated GitLab/local CI helpers for EDW patch verification and ParseBench. |
| `edw_rag_patches/tests/` | EDW patch regression tests. |
| `examples/edw_rag_demo.py` | Pipeline comparison demo (with/without patches). |
| `.env` | GPUStack configuration for Qwen models. |
| `.gitlab-ci.yml` | CI/CD pipeline with patch integrity checks. |

## Requirements

- Python 3.10+
- LightRAG (local clone with `uv sync --extra api`)
- GPUStack (or any OpenAI-compatible API) for LLM / embedding / reranker
- Docling (for PDF parsing with bounding-box data)
