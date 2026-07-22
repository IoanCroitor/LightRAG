#!/usr/bin/env python3
"""Generate per-document ground-truth Q&A from the ParseBench corpus.

For every PDF in :data:`config.CORPUS_DIR` we:
  1. extract its text (see :mod:`pdf_text`),
  2. ask the LLM to produce ``N`` factual question/answer pairs that are
     directly answerable from the excerpt, each with a supporting snippet,
  3. validate the JSON and write one file per document to
     :data:`config.GROUND_TRUTH_DIR/<doc_id>.json`.

These files are the "predefined JSON" the evaluation suite later compares
against the RAG system's answers via an LLM judge.

Usage:
    uv run python edw_parsebench_eval/build_ground_truth.py [--limit N] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import config
import llm_client
import pdf_text

logger = logging.getLogger("parsebench.gt")

SYSTEM_PROMPT = (
    "You are a careful dataset annotator for a retrieval-augmented generation "
    "(RAG) benchmark. You are given an excerpt of a document. Your job is to "
    "write factual question/answer pairs that a RAG system should be able to "
    "answer correctly using ONLY the information in the excerpt."
)

USER_TEMPLATE = """\
Document excerpt (truncated):
---
{text}
---

Write exactly {n} question/answer pairs that are:
- Factual and directly answerable from the excerpt above (do not use outside knowledge).
- Diverse: cover different parts/facts of the document, not near-duplicates.
- Answered with a concise phrase or short sentence (the "answer"), not a long paragraph.
- Each paired with a short "snippet": a verbatim quote (<= 200 chars) from the excerpt that supports the answer.

Return ONLY a JSON array, no prose, no markdown fences:
[
  {{"question": "...", "answer": "...", "snippet": "..."}},
  ...
]

If the excerpt is too short or garbled to support {n} distinct factual pairs,
return as many as you can (at least 1) and nothing else.
"""


def _validate(items: list[dict]) -> list[dict]:
    clean: list[dict] = []
    for i, it in enumerate(items, 1):
        if not isinstance(it, dict):
            continue
        q = (it.get("question") or "").strip()
        a = (it.get("answer") or "").strip()
        if not q or not a:
            continue
        clean.append(
            {
                "id": f"q{i}",
                "question": q,
                "answer": a,
                "snippet": (it.get("snippet") or "").strip(),
            }
        )
    return clean


def generate_for_pdf(pdf_path: Path, n: int) -> dict:
    text, page_count = pdf_text.extract_pdf_text(pdf_path, config.GT_MAX_CONTEXT_CHARS)
    doc_id = pdf_path.stem
    out: dict = {
        "doc_id": doc_id,
        "source_file": pdf_path.name,
        "num_pages": page_count,
        "generator": {
            "model": config.LLM_MODEL,
            "questions_requested": n,
            "extracted_chars": len(text),
        },
        "questions": [],
    }
    if not text.strip():
        logger.warning("%s: no extractable text; skipping QA generation", pdf_path.name)
        return out

    last_err: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            data = llm_client.chat_json(
                SYSTEM_PROMPT,
                USER_TEMPLATE.format(text=text, n=n),
                model=config.LLM_MODEL,
                temperature=config.GT_TEMPERATURE,
                max_tokens=4096,
            )
            if isinstance(data, dict):
                # Some models wrap the array under a key.
                for key in ("questions", "qa", "pairs", "data"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
            if not isinstance(data, list):
                raise ValueError(f"expected list, got {type(data).__name__}")
            out["questions"] = _validate(data)
            if out["questions"]:
                logger.info(
                    "%s: generated %d Q&A pairs", pdf_path.name, len(out["questions"])
                )
                return out
            last_err = ValueError("model returned 0 valid pairs")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("%s: attempt %d failed: %s", pdf_path.name, attempt, exc)
    logger.error("%s: giving up after %d attempts (%s)", pdf_path.name, config.MAX_RETRIES, last_err)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only the first N PDFs")
    ap.add_argument("--overwrite", action="store_true", help="regenerate existing files")
    ap.add_argument("--questions-per-doc", type=int, default=config.QUESTIONS_PER_DOC)
    args = ap.parse_args()

    config.GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(config.CORPUS_DIR.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]

    if not pdfs:
        logger.error("No PDFs found in %s", config.CORPUS_DIR)
        return 1

    total = 0
    for pdf in pdfs:
        out_path = config.GROUND_TRUTH_DIR / f"{pdf.stem}.json"
        if out_path.exists() and not args.overwrite:
            logger.info("%s: ground truth exists, skipping (use --overwrite)", pdf.name)
            continue
        result = generate_for_pdf(pdf, args.questions_per_doc)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        total += len(result["questions"])

    logger.info("Done. %d Q&A pairs written across %d documents.", total, len(pdfs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
