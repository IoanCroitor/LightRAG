#!/usr/bin/env python3
"""Upload the ParseBench corpus into a running LightRAG server.

Usage:
    # Clear existing data, upload via /documents/upload, wait until idle:
    uv run python edw_parsebench_eval/ingest.py [--clear] [--scan]

The default strategy uploads each PDF file individually via ``/documents/upload``
(HTTP multipart) and then polls until every document is PROCESSED.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config
from api_client import LightRAGClient

logger = logging.getLogger("parsebench.ingest")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", action="store_true", help="clear all documents before upload")
    ap.add_argument("--scan", action="store_true", help="use /documents/scan instead of per-file upload")
    ap.add_argument("--timeout", type=float, default=1800.0, help="max seconds to wait for idle")
    ap.add_argument("--base-url", help="override LIGHTRAG_BASE_URL")
    args = ap.parse_args()

    client = LightRAGClient(base_url=args.base_url)

    if args.clear:
        logger.info("Clearing all documents ...")
        try:
            resp = client.clear_documents()
            logger.info("Clear: %s", resp.get("message", "ok"))
        except Exception as exc:
            logger.warning("Clear failed (may be already idle): %s", exc)

    if args.scan:
        # Use scan: copy files to the server's INPUT_DIR first, then trigger scan.
        logger.info("Triggering /documents/scan ...")
        resp = client.scan()
        logger.info("Scan: %s (track_id=%s)", resp.get("message", "ok"), resp.get("track_id", "?"))
    else:
        # Upload each file individually.
        pdfs = sorted(config.CORPUS_DIR.glob("*.pdf"))
        if not pdfs:
            logger.error("No PDFs found in %s", config.CORPUS_DIR)
            return 1
        logger.info("Uploading %d documents ...", len(pdfs))
        for pdf in pdfs:
            try:
                resp = client.upload_file(pdf)
                track_id = resp.get("track_id", "?")
                logger.info("  %s -> %s [%s]", pdf.name, resp.get("status", "?"), track_id)
            except Exception as exc:
                logger.error("  %s FAILED: %s", pdf.name, exc)
                continue

    logger.info("Waiting for pipeline to become idle (timeout=%ss) ...", args.timeout)
    try:
        counts = client.wait_until_idle(timeout=args.timeout)
        logger.info("Pipeline idle. Status counts: %s", counts)
        processed = sum(counts.get(k, 0) for k in ("PROCESSED",))
        failed = counts.get("FAILED", 0)
        if failed:
            logger.warning("%d document(s) FAILED. Check /documents/paginated for details.", failed)
        logger.info("Ingestion complete: %d documents PROCESSED.", processed)
    except TimeoutError as exc:
        logger.error(exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
