#!/usr/bin/env python3
"""Evaluate RAG system answers against ground-truth Q&A via an LLM judge.

For each ground-truth document JSON:
  1. POST ``/query`` to the LightRAG server for every question.
  2. Ask the judge LLM whether the RAG response matches the ground-truth answer.
  3. Optionally also query ``/query/data`` to verify the EDW-RAG
     ``citation_highlights`` patch is active.
  4. Aggregate per-document and overall accuracy.

Usage:
    uv run python edw_parsebench_eval/run_eval.py [--no-citation-check] [--output-dir ...]

Outputs:
    ``results/detailed.json``   — per-question judge verdicts + metadata
    ``results/summary.txt``     — human-readable overview
"""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import config
from api_client import LightRAGClient
import llm_client

logger = logging.getLogger("parsebench.eval")


def _configure_logging(log_path: Path, level_name: str) -> None:
    """Send timestamped evaluation logs to both the console and a file."""
    level = getattr(logging, level_name.upper())
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[console, file_handler], force=True)


def _shorten(value: Any, limit: int = 180) -> str:
    """Return a single-line, bounded diagnostic value for the log."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[:limit - 1]}…"


def _score(value: Any) -> float:
    """Normalize judge scores so one malformed verdict cannot stop the run."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are an impartial judge evaluating a Retrieval-Augmented Generation "
    "(RAG) system. Your task is to determine whether the RAG system's response "
    "is factually correct given the question and the reference (ground-truth) "
    "answer from the source document."
)

JUDGE_TEMPLATE = """\
Question: {question}

Reference answer (ground truth, from source document):
{reference}

RAG system response:
{rag_response}

---

Decide whether the RAG response correctly answers the question, considering:
- The RAG response does NOT need to match the reference answer word-for-word.
- It IS correct if it conveys the same factual information or a superset of it.
- It is INCORRECT if it contradicts the reference answer, hallucinates facts
  not supported by the reference, or fails to answer the question at all.
- If the reference answer is numeric or has a specific entity, the RAG response
  must get that number/entity right to be considered correct.

Return ONLY a JSON object (no prose, no fences):
{{
  "correct": true/false,
  "score": <float 0.0-1.0>,
  "rationale": "<short explanation>"
}}
"""

# ---------------------------------------------------------------------------
# Citation-highlights probe (EDW-RAG patch feature)
# ---------------------------------------------------------------------------

CITATION_PROBE_SYSTEM = "You are a test probe checking if citation_highlights appear in a RAG data response."

CITATION_PROBE_QUERIES = [
    "What is the main topic of this document?",
    "Summarize the key facts in this document.",
    "What entities and relationships are mentioned?",
]


def _check_citation_highlights(client: LightRAGClient) -> dict:
    """Best-effort probe for the EDW-RAG citation_highlights patch."""
    result: dict = {"present": False, "version_seen": None, "details": ""}
    for probe_idx, q in enumerate(CITATION_PROBE_QUERIES, 1):
        started = time.perf_counter()
        try:
            logger.info(
                "Citation probe %d/%d: %s",
                probe_idx,
                len(CITATION_PROBE_QUERIES),
                _shorten(q),
            )
            resp = client.query_data(q, mode="mix", include_citation_highlights=True)
            data = resp.get("data", {})
            elapsed = time.perf_counter() - started
            if not data:
                logger.warning(
                    "Citation probe %d/%d returned no data (%.2fs).",
                    probe_idx,
                    len(CITATION_PROBE_QUERIES),
                    elapsed,
                )
                continue
            if "citation_highlights" in data:
                ch = data["citation_highlights"]
                result["present"] = True
                result["version_seen"] = ch.get("version")
                result["details"] = f"Found citation_highlights v{result['version_seen']} in {len(ch.get('sources', {}))} source(s)."
                result["probe"] = {"index": probe_idx, "query": q, "elapsed_s": round(elapsed, 3)}
                logger.info("Citation probe succeeded in %.2fs: %s", elapsed, result["details"])
                return result
            # Check in nested structure
            for key in ("response",):
                if isinstance(data.get(key), dict) and "citation_highlights" in data[key]:
                    ch = data[key]["citation_highlights"]
                    result["present"] = True
                    result["version_seen"] = ch.get("version")
                    result["details"] = f"Found inside response envelope."
                    result["probe"] = {"index": probe_idx, "query": q, "elapsed_s": round(elapsed, 3)}
                    logger.info("Citation probe succeeded in %.2fs: %s", elapsed, result["details"])
                    return result
        except Exception as exc:
            result["details"] = f"Probe failed: {exc}"
            logger.warning(
                "Citation probe %d/%d failed after %.2fs: %s",
                probe_idx,
                len(CITATION_PROBE_QUERIES),
                time.perf_counter() - started,
                exc,
            )
            break
    if not result["present"]:
        if not result["details"]:
            result["details"] = "No citation_highlights found in any probe query."
    return result


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------


def judge(question: str, reference: str, rag_response: str) -> dict:
    """Ask the LLM judge to compare RAG response against reference answer."""
    user = JUDGE_TEMPLATE.format(
        question=question,
        reference=reference,
        rag_response=rag_response or "(empty)",
    )
    try:
        result = llm_client.chat_json(
            JUDGE_SYSTEM,
            user,
            model=config.JUDGE_MODEL,
            temperature=config.JUDGE_TEMPERATURE,
            max_tokens=1024,
        )
        if isinstance(result, dict):
            # Normalise keys.
            if "correct" not in result and "score" in result:
                result["correct"] = result.get("score", 0) >= 0.5
            result.setdefault("correct", False)
            result.setdefault("score", 0.0)
            result.setdefault("rationale", "")
            return result
        return {"correct": False, "score": 0.0, "rationale": f"Unexpected judge format: {type(result).__name__}"}
    except Exception as exc:
        logger.warning("Judge call failed: %s", exc)
        return {"correct": False, "score": 0.0, "rationale": f"Judge error: {exc}"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-citation-check", action="store_true", help="skip the EDW-RAG probe")
    ap.add_argument("--output-dir", type=Path, default=config.RESULTS_DIR)
    ap.add_argument(
        "--log-file",
        type=Path,
        help="write detailed run logs here (default: <output-dir>/evaluation.log)",
    )
    ap.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="minimum log level for console and file output (default: INFO)",
    )
    ap.add_argument("--max-questions", type=int, default=0, help="limit total questions (for testing)")
    ap.add_argument(
        "--workers",
        "-w",
        type=int,
        default=config.EVAL_MAX_WORKERS,
        help="number of parallel worker threads for querying and judging",
    )
    args = ap.parse_args()

    RESULTS_DIR = args.output_dir
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or RESULTS_DIR / "evaluation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_logging(log_path, args.log_level)

    run_started_at = datetime.now(timezone.utc).isoformat()
    run_started = time.perf_counter()
    logger.info("Evaluation run started at %s", run_started_at)
    logger.info(
        "Configuration: server=%s mode=%s workers=%d judge_model=%s top_k=%s chunk_top_k=%s rerank=%s citation_highlights=%s",
        config.LIGHTRAG_BASE_URL,
        config.QUERY_MODE,
        max(1, args.workers),
        config.JUDGE_MODEL,
        config.QUERY_TOP_K,
        config.QUERY_CHUNK_TOP_K,
        config.QUERY_ENABLE_RERANK,
        config.INCLUDE_CITATION_HIGHLIGHTS,
    )
    logger.info("Detailed log: %s", log_path)

    client = LightRAGClient()

    # Load ground truth
    gt_files = sorted(config.GROUND_TRUTH_DIR.glob("*.json"))
    if not gt_files:
        logger.error("No ground-truth files in %s", config.GROUND_TRUTH_DIR)
        return 1

    all_qa: list[dict] = []
    doc_questions: dict[str, int] = {}
    for gt_path in gt_files:
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        doc_id = gt.get("doc_id", gt_path.stem)
        for qa in gt.get("questions", []):
            all_qa.append({"doc_id": doc_id, **qa})
        doc_questions[doc_id] = len(gt.get("questions", []))
    logger.info(
        "Loaded %d ground-truth Q&A pairs across %d documents.",
        len(all_qa),
        len(doc_questions),
    )

    if args.max_questions:
        all_qa = all_qa[: args.max_questions]
        logger.info("Limited to %d questions (--max-questions).", args.max_questions)

    # Check citation highlights (EDW-RAG patch)
    citation_info: dict[str, Any] = {"skipped": args.no_citation_check}
    if not args.no_citation_check:
        logger.info("Probing for EDW-RAG citation_highlights ...")
        citation_info = _check_citation_highlights(client)
        logger.info("Citation highlights: %s", citation_info.get("details"))

    # Run evaluation
    # Run evaluation (in parallel if workers > 1)
    results_by_idx: dict[int, dict] = {}
    correct_count = 0
    total_score = 0.0
    completed_count = 0
    n = len(all_qa)
    workers = max(1, args.workers)
    lock = threading.Lock()

    thread_local = threading.local()

    def _get_worker_client() -> LightRAGClient:
        if not hasattr(thread_local, "client"):
            thread_local.client = LightRAGClient()
        return thread_local.client

    def _eval_one(idx: int, item: dict) -> tuple[int, dict, bool, float]:
        question = item["question"]
        reference = item["answer"]
        c = _get_worker_client()
        question_id = item.get("id", f"q{idx}")
        started = time.perf_counter()
        query_elapsed = 0.0
        judge_elapsed = 0.0
        query_error: str | None = None

        logger.info(
            "[%d/%d] START doc=%s question=%s text=%s",
            idx,
            n,
            item["doc_id"],
            question_id,
            _shorten(question),
        )

        try:
            # Query the RAG system
            query_started = time.perf_counter()
            response = c.query(question, include_references=True)
            query_elapsed = time.perf_counter() - query_started
            rag_text = response.get("response", "")
            refs = response.get("references", [])
        except Exception as exc:
            query_elapsed = time.perf_counter() - started
            query_error = f"{type(exc).__name__}: {exc}"
            rag_text = ""
            refs = []
            logger.warning("[%d/%d] Query failed after %.2fs: %s", idx, n, query_elapsed, query_error)

        # Judge
        judge_started = time.perf_counter()
        verdict = judge(question, reference, rag_text)
        judge_elapsed = time.perf_counter() - judge_started

        is_correct = bool(verdict.get("correct", False))
        score = _score(verdict.get("score"))
        verdict["correct"] = is_correct
        verdict["score"] = score
        total_elapsed = time.perf_counter() - started

        entry: dict = {
            "doc_id": item["doc_id"],
            "question_id": question_id,
            "question": question,
            "reference_answer": reference,
            "rag_response": rag_text,
            "snippet": item.get("snippet", ""),
            "judge_verdict": verdict,
            "references": refs,
            "timing": {
                "query_s": round(query_elapsed, 3),
                "judge_s": round(judge_elapsed, 3),
                "total_s": round(total_elapsed, 3),
            },
            "query_error": query_error,
        }
        logger.info(
            "[%d/%d] DONE doc=%s question=%s correct=%s score=%.3f query=%.2fs judge=%.2fs total=%.2fs answer_chars=%d references=%d rationale=%s",
            idx,
            n,
            item["doc_id"],
            question_id,
            bool(is_correct),
            score,
            query_elapsed,
            judge_elapsed,
            total_elapsed,
            len(rag_text),
            len(refs) if isinstance(refs, list) else 0,
            _shorten(verdict.get("rationale")),
        )
        return idx, entry, is_correct, score

    logger.info("Evaluating %d questions with %d worker(s) (judge model: %s) ...", n, workers, config.JUDGE_MODEL)

    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_eval_one, idx, item)
                for idx, item in enumerate(all_qa, 1)
            ]
            for future in concurrent.futures.as_completed(futures):
                idx, entry, is_correct, score = future.result()
                with lock:
                    results_by_idx[idx] = entry
                    completed_count += 1
                    if is_correct:
                        correct_count += 1
                    total_score += score

                    if completed_count % 10 == 0 or completed_count == n:
                        logger.info(
                            "[%d/%d] accuracy=%.0f%% avg_score=%.2f",
                            completed_count,
                            n,
                            correct_count / completed_count * 100,
                            total_score / completed_count,
                        )
    else:
        for idx, item in enumerate(all_qa, 1):
            idx, entry, is_correct, score = _eval_one(idx, item)
            results_by_idx[idx] = entry
            completed_count += 1
            if is_correct:
                correct_count += 1
            total_score += score

            if completed_count % 10 == 0 or completed_count == n:
                logger.info(
                    "[%d/%d] accuracy=%.0f%% avg_score=%.2f",
                    completed_count,
                    n,
                    correct_count / completed_count * 100,
                    total_score / completed_count,
                )

    results = [results_by_idx[i] for i in range(1, n + 1)]
    # Aggregate
    run_elapsed = time.perf_counter() - run_started
    accuracy = (correct_count / n * 100) if n else 0.0
    avg_score = (total_score / n) if n else 0.0
    query_error_count = sum(bool(result.get("query_error")) for result in results)

    # Per-doc breakdown
    doc_stats: dict[str, dict] = {}
    for r in results:
        doc = r["doc_id"]
        if doc not in doc_stats:
            doc_stats[doc] = {"total": 0, "correct": 0, "score_sum": 0.0}
        doc_stats[doc]["total"] += 1
        if r["judge_verdict"].get("correct"):
            doc_stats[doc]["correct"] += 1
        doc_stats[doc]["score_sum"] += r["judge_verdict"].get("score", 0.0)

    doc_summary: list[dict] = []
    for doc, st in sorted(doc_stats.items()):
        doc_summary.append(
            {
                "doc_id": doc,
                "total": st["total"],
                "correct": st["correct"],
                "accuracy_pct": round(st["correct"] / st["total"] * 100, 1),
                "avg_score": round(st["score_sum"] / st["total"], 3),
            }
        )

    summary = {
        "total_questions": n,
        "total_documents": len(doc_stats),
        "correct_count": correct_count,
        "accuracy_pct": round(accuracy, 1),
        "avg_score": round(avg_score, 3),
        "judge_model": config.JUDGE_MODEL,
        "query_mode": config.QUERY_MODE,
        "citation_highlights": citation_info,
        "query_error_count": query_error_count,
        "elapsed_s": round(run_elapsed, 3),
        "documents": doc_summary,
    }

    # Write detailed results
    detailed_path = RESULTS_DIR / "detailed.json"
    detailed: dict = {
        "config": {
            "query_mode": config.QUERY_MODE,
            "judge_model": config.JUDGE_MODEL,
            "llm_model": config.LLM_MODEL,
            "top_k": config.QUERY_TOP_K,
            "chunk_top_k": config.QUERY_CHUNK_TOP_K,
            "enable_rerank": config.QUERY_ENABLE_RERANK,
            "workers": workers,
            "server": config.LIGHTRAG_BASE_URL,
            "run_started_at": run_started_at,
            "log_file": str(log_path),
        },
        "citation_highlights": citation_info,
        "summary": summary,
        "results": results,
    }
    detailed_path.write_text(json.dumps(detailed, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write summary
    summary_path = RESULTS_DIR / "summary.txt"
    lines = [
        "=" * 55,
        "ParseBench RAG Evaluation Summary",
        "=" * 55,
        f"  Questions evaluated:  {n}",
        f"  Documents tested:     {len(doc_stats)}",
        f"  Query mode:           {config.QUERY_MODE}",
        f"  Judge model:          {config.JUDGE_MODEL}",
        f"  Correct:              {correct_count} / {n}",
        f"  Accuracy:             {accuracy:.1f}%",
        f"  Average score:        {avg_score:.3f}",
        f"  Query errors:         {query_error_count}",
        f"  Elapsed:              {run_elapsed:.1f}s",
        "",
        "Per-document breakdown:",
        f"  {'Doc ID':<35s} {'Q':>3s} {'OK':>4s} {'Acc':>5s} {'Score':>6s}",
        "  " + "-" * 55,
    ]
    for ds in doc_summary:
        lines.append(
            f"  {ds['doc_id']:<35s} {ds['total']:>3d} {ds['correct']:>4d}"
            f" {ds['accuracy_pct']:>5.1f}% {ds['avg_score']:>6.3f}"
        )
    lines.append("")
    if citation_info.get("present"):
        lines.append(f"  EDW-RAG citation_highlights detected: v{citation_info.get('version_seen')}")
    else:
        lines.append(f"  EDW-RAG citation_highlights: {citation_info.get('details','not checked')}")
    lines.append("")
    lines.append(f"  Detailed results:    {detailed_path}")
    lines.append("=" * 55)

    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text, encoding="utf-8")
    logger.info(
        "Evaluation completed: accuracy=%.1f%% avg_score=%.3f errors=%d elapsed=%.2fs. Results: %s",
        accuracy,
        avg_score,
        query_error_count,
        run_elapsed,
        detailed_path,
    )
    print("\n" + summary_text)

    return 0 if accuracy >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
