#!/usr/bin/env python3
"""EDW-RAG Patches Demo / Test.

Compares LightRAG query responses WITH and WITHOUT the chunk-level citation
highlight patches, using GPUStack-hosted Qwen models.

Usage:
    uv run python examples/edw_rag_demo.py [--llm MODEL] [--embed MODEL]
                                          [--clean] [--keep]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# GPUStack provider config
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("GPU_STACK_API", "http://fdz2.edw.ro/v1")
API_KEY = os.environ.get(
    "GPU_STACK_API_KEY",
    "gpustack_a02575b7057c5544_4410055a6258ec30d278f5b9ec3ba950",
)
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b-fast")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding-0.6b")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "qwen3-reranker-0.6b")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))
def make_llm_func():
    """OpenAI-compatible LLM wrapper -> GPUStack Qwen.

    The Qwen reasoning model spends 4K+ tokens on reasoning before producing
    content.  We enforce a minimum ``max_tokens`` of 8192 so entity extraction
    and other pipeline calls don't run out of budget.
    """
    from lightrag.llm.openai import openai_complete_if_cache

    _MIN_MAX_TOKENS = 8192

    async def _llm(prompt, system_prompt=None, history_messages=None, **kw):
        # Ensure enough token budget for reasoning + content
        mt = kw.pop("max_tokens", None)
        if mt is None or mt < _MIN_MAX_TOKENS:
            mt = _MIN_MAX_TOKENS
        return await openai_complete_if_cache(
            LLM_MODEL, prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=API_BASE, api_key=API_KEY,
            max_tokens=mt, **kw,
        )
    return _llm

def make_embed_func():
    """OpenAI-compatible embedding -> GPUStack Qwen (1024-dim).

    Calls the raw ``openai_embed.func`` (without its 1536-dim decorator)
    and wraps it with our own 1024-dim dimension annotation.
    """
    import numpy as np
    from lightrag.llm.openai import openai_embed as _openai_embed_fn
    from lightrag.utils import wrap_embedding_func_with_attrs

    # Use the undecorated function (strip @wrap_embedding_func_with_attrs
    # but keep @retry).  ``.func`` is the next layer in the chain.
    _raw_embed = getattr(_openai_embed_fn, "func", _openai_embed_fn)

    @wrap_embedding_func_with_attrs(
        embedding_dim=EMBED_DIM, max_token_size=8192, model_name=EMBED_MODEL,
    )
    async def _embed(texts: list[str]) -> np.ndarray:
        return await _raw_embed(
            texts, model=EMBED_MODEL,
            base_url=API_BASE, api_key=API_KEY,
        )
    return _embed
def make_rerank_func():
    """GPUStack reranker."""
    import aiohttp
    from lightrag.utils import logger

    async def _rerank(query: str, chunks: list[dict]) -> list[dict]:
        docs = [c.get("content", "") for c in chunks]
        if not docs:
            return chunks
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{API_BASE}/rerank",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={"model": RERANK_MODEL, "query": query,
                          "documents": docs, "top_n": len(docs)},
                ) as r:
                    if r.status == 200:
                        data = await r.json()
                        scored = [(x["index"], x["relevance_score"])
                                  for x in data.get("results", [])]
                        scored.sort(key=lambda x: -x[1])
                        return [chunks[i] for i, _ in scored]
        except Exception as e:
            logger.warning(f"Reranker failed: {e}")
        return chunks
    return _rerank


# ---------------------------------------------------------------------------
# Test documents
# ---------------------------------------------------------------------------

DOCUMENTS = [
    textwrap.dedent("""\
    # Artificial Intelligence Overview

    Artificial Intelligence (AI) is a branch of computer science that aims to create
    intelligent machines capable of performing tasks that typically require human
    intelligence. These tasks include learning, reasoning, problem-solving, perception,
    and language understanding.

    ## History of AI

    The field of AI was formally founded in 1956 at a conference at Dartmouth College.
    Early pioneers included Alan Turing, John McCarthy, Marvin Minsky, and others.
    Since then, AI has gone through several "summers" and "winters" of funding and
    interest.

    ## Machine Learning

    Machine learning is a subset of AI that enables systems to learn and improve from
    experience without being explicitly programmed. Deep learning, a further subset,
    uses neural networks with many layers to model complex patterns in data.

    ## Applications

    AI is used in many industries including healthcare (diagnosis), finance (trading),
    transportation (autonomous vehicles), and entertainment (recommendation systems).
    """),
    textwrap.dedent("""\
    # Climate Change Report

    Climate change refers to long-term shifts in temperatures and weather patterns.
    Human activities, especially the burning of fossil fuels, have been the main
    driver of climate change since the 1800s.

    ## Greenhouse Gases

    The primary greenhouse gases include carbon dioxide (CO2), methane (CH4),
    and nitrous oxide (N2O). CO2 levels have increased by over 50% since the
    Industrial Revolution, from 280 ppm to over 420 ppm today.

    ## Global Temperature Rise

    The Earth's average temperature has risen by approximately 1.2°C since the
    late 19th century. Most of this warming has occurred in the last 40 years.
    The Arctic is warming about four times faster than the global average.

    ## Mitigation Strategies

    Key strategies include transitioning to renewable energy, improving energy
    efficiency, reforestation, and developing carbon capture technologies.
    The Paris Agreement, adopted in 2015, aims to limit global warming to well
    below 2°C above pre-industrial levels.
    """),
]

QUERIES = [
    "What is machine learning and how does it relate to AI?",
    "What are the main causes and effects of climate change?",
]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

WORK_DIR = Path("./rag_storage_demo")


async def run_phase(label: str, use_patches: bool, clean: bool) -> dict:
    """Run one test phase (with or without patches). Returns result stats."""
    from lightrag import LightRAG, QueryParam

    if clean and WORK_DIR.exists():
        shutil.rmtree(str(WORK_DIR))
    rag = LightRAG(
        working_dir=str(WORK_DIR),
        llm_model_func=make_llm_func(),
        embedding_func=make_embed_func(),
        addon_params={
            "enable_rerank": True,
            "chunk_token_size": 1024,
            "chunk_overlap_token_size": 128,
        },
        log_level="WARNING",
    )
    await rag.initialize_storages()

    # Inject reranker
    rag._rerank = make_rerank_func()

    # Ingest
    t0 = time.perf_counter()
    for doc in DOCUMENTS:
        await rag.ainsert(doc)
    ingest_t = time.perf_counter() - t0
    print(f"  Ingested {len(DOCUMENTS)} docs in {ingest_t:.1f}s")
    # Query
    results = []
    for q in QUERIES:
        t0 = time.perf_counter()
        result = await rag.aquery_llm(
            q,
            param=QueryParam(
                mode="mix", top_k=10,
                include_references=True,
            ),
        )
        qt = time.perf_counter() - t0
        data = result.get("data", {})
        results.append({
            "query": q,
            "time_s": round(qt, 2),
            "response": (result.get("llm_response", {}) or {}).get("content", "")[:200],
            "n_references": len(data.get("references", [])),
            "n_chunks_with_positions": sum(
                1 for c in data.get("chunks", []) if c.get("positions")
            ),
            "total_chunks": len(data.get("chunks", [])),
            "has_highlights": "citation_highlights" in data,
            "highlights": data.get("citation_highlights"),
        })

    await rag.finalize_storages()
    return {"label": label, "use_patches": use_patches, "results": results}


async def main(clean: bool, keep: bool):
    print("=" * 72)
    print("EDW-RAG Patches  —  Chunk-Level Citation Demo")
    print("=" * 72)
    print(f"LLM: {LLM_MODEL}")
    print(f"Embed: {EMBED_MODEL}  (dim={EMBED_DIM})")
    print(f"Rerank: {RERANK_MODEL}")
    print()

    # ── Phase 1: WITHOUT patches ──────────────────────────────────
    print("─── Phase 1: WITHOUT patches ───")
    no_patch = await run_phase("no-patches", use_patches=False, clean=clean)
    print()

    # ── Phase 2: WITH patches ─────────────────────────────────────
    print("─── Phase 2: WITH patches ───")
    with_patch = await run_phase("with-patches", use_patches=True, clean=True)
    print()

    # ── Report ────────────────────────────────────────────────────
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    for phase in [no_patch, with_patch]:
        print(f"\n{phase['label']}:")
        for r in phase["results"]:
            print(f"  Query: {r['query'][:60]}...")
            print(f"    Time:     {r['time_s']}s")
            print(f"    Refs:     {r['n_references']}")
            print(f"    Chunks:   {r['n_chunks_with_positions']}/{r['total_chunks']} with positions")
            print(f"    Sidecar:  {r['has_highlights']}")
            if r.get("highlights"):
                n_sources = len(r["highlights"].get("sources", {}))
                print(f"    Sources:  {n_sources}")
                for fp, src in r["highlights"]["sources"].items():
                    n_hl = sum(len(ch.get("highlights", [])) for ch in src.get("chunks", []))
                    print(f"      {fp}: {len(src['chunks'])} chunks, {n_hl} highlights")
            print(f"    Response: {r['response'][:100]}...")

    # ── Verify ────────────────────────────────────────────────────
    no_has = any(r.get("has_highlights") for r in no_patch["results"])
    w_has = any(r.get("has_highlights") for r in with_patch["results"])

    print(f"\n{'=' * 72}")
    if w_has and not no_has:
        print("VERDICT: Patches successfully added chunk-level citation highlights.")
        print("         Document-level citations work in both modes.")
        print("         Chunk-level bounding-box highlights only appear with patches.")
    elif w_has and no_has:
        print("VERDICT: Citation highlights present in both modes (unexpected).")
    else:
        print("VERDICT: Citation highlights not detected. See details above.")
        print("         (Note: positions appear only when Docling blocks_path is")
        print("          available during chunking, which requires Docling parsing.")

    # ── Sidecar JSON preview ──────────────────────────────────────
    if keep:
        for r in with_patch["results"]:
            if r.get("highlights"):
                out = Path(f"citation_highlights_{r['query'][:20].replace(' ', '_')}.json")
                out.write_text(json.dumps(r["highlights"], indent=2))
                print(f"  Wrote {out}")

    print()

    # Cleanup unless --keep
    if not keep and WORK_DIR.exists():
        shutil.rmtree(str(WORK_DIR))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default=LLM_MODEL)
    p.add_argument("--embed", default=EMBED_MODEL)
    p.add_argument("--clean", action="store_true", default=True)
    p.add_argument("--keep", action="store_true", help="keep working dir")
    p.add_argument("--output", "-o", help="save sidecar JSON to path")
    args = p.parse_args()
    asyncio.run(main(clean=args.clean, keep=args.keep))
