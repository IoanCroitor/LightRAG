#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ParseBench E2E evaluation: build ground truth → ingest → query + judge.
#
# Prerequisites:
#   1. LightRAG server is running (e.g. `uv run python -m edw_rag_patches.cli`)
#   2. .env is present in the repo root (LLM credentials, server config)
#   3. Corollary PDFs are already in edw_parsebench_eval/corpus/
#
# Steps:
#   build-gt   – (re)generate per-document ground-truth Q&A via the LLM
#   ingest     – upload corpus to the server, wait until pipeline is idle
#   evaluate   – run every question through /query, judge vs ground truth
#   all        – run all three steps in order
#
# Usage:
#   cd ~/Documents/edw_rag/LightRAG
#   bash edw_parsebench_eval/run_all.sh [command]
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root

SCRIPT_DIR="$(dirname "$0")"
BIN="uv run python"

case "${1:-all}" in
  build-gt)
    echo "=== Build ground truth ==="
    $BIN "$SCRIPT_DIR/build_ground_truth.py" --questions-per-doc 6 --overwrite
    ;;
  ingest)
    echo "=== Ingest corpus ==="
    $BIN "$SCRIPT_DIR/ingest.py"
    ;;
  evaluate)
    echo "=== Evaluate ==="
    $BIN "$SCRIPT_DIR/run_eval.py"
    ;;
  all)
    $0 build-gt
    $0 ingest
    $0 evaluate
    ;;
  *)
    echo "Usage: $0 [build-gt|ingest|evaluate|all]"
    exit 1
    ;;
esac
