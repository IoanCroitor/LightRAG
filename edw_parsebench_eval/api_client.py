"""Thin synchronous HTTP client for the LightRAG REST API.

Wraps the endpoints the evaluation suite needs: ingestion
(``/documents/upload`` + ``/documents/scan``), status polling
(``/documents/pipeline_status``, ``/documents/status_counts``,
``/documents/track_status/{id}``), and querying (``/query`` for the generated
answer, ``/query/data`` for retrieval context + EDW-RAG citation highlights).

Auth is optional (the deployment under test is fully open), supplied via the
``X-API-Key`` header from :data:`config.LIGHTRAG_API_KEY`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import httpx

import config


class LightRAGClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or config.LIGHTRAG_BASE_URL).rstrip("/")
        self._headers = config.auth_headers()
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._headers,
            timeout=config.HTTP_TIMEOUT,
            follow_redirects=True,
        )

    # -- low level ---------------------------------------------------------

    def _post(self, path: str, **kwargs) -> httpx.Response:
        return self._client.post(path, **kwargs)

    def _get(self, path: str, **kwargs) -> httpx.Response:
        return self._client.get(path, **kwargs)

    # -- ingestion ---------------------------------------------------------

    def upload_file(self, pdf_path: Path) -> dict:
        """Upload a single file to the input dir and enqueue it for indexing."""
        pdf_path = Path(pdf_path)
        with pdf_path.open("rb") as fh:
            files = {"file": (pdf_path.name, fh, "application/pdf")}
            resp = self._post(
                "/documents/upload",
                files=files,
                timeout=config.UPLOAD_TIMEOUT,
            )
        resp.raise_for_status()
        return resp.json()

    def scan(self) -> dict:
        """Trigger a scan of the input directory (alternative to upload)."""
        resp = self._post("/documents/scan")
        resp.raise_for_status()
        return resp.json()

    def clear_documents(self) -> dict:
        resp = self._delete("/documents")
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str, **kwargs) -> httpx.Response:
        return self._client.request("DELETE", path, **kwargs)

    # -- status ------------------------------------------------------------

    def pipeline_status(self) -> dict:
        resp = self._get("/documents/pipeline_status")
        resp.raise_for_status()
        return resp.json()

    def status_counts(self) -> dict:
        resp = self._get("/documents/status_counts")
        resp.raise_for_status()
        return resp.json()

    def track_status(self, track_id: str) -> dict:
        resp = self._get(f"/documents/track_status/{track_id}")
        resp.raise_for_status()
        return resp.json()

    def is_idle(self) -> bool:
        """True when the pipeline is not busy and nothing is pending/failed."""
        try:
            status = self.pipeline_status()
        except Exception:  # noqa: BLE001
            return False
        if status.get("busy"):
            return False
        try:
            counts = self.status_counts().get("status_counts", {})
        except Exception:  # noqa: BLE001
            return False
        active = sum(
            counts.get(k, 0)
            for k in ("PENDING", "PROCESSING", "PREPROCESSED")
        )
        failed = counts.get("FAILED", 0)
        return active == 0 and failed == 0

    def wait_until_idle(self, timeout: float = 1800.0, poll: float = 5.0) -> dict:
        """Block until ``is_idle()`` or timeout. Returns the final status_counts."""
        deadline = time.time() + timeout
        last_counts: dict = {}
        while time.time() < deadline:
            try:
                last_counts = self.status_counts().get("status_counts", {})
            except Exception:  # noqa: BLE001
                last_counts = {}
            if self.is_idle():
                return last_counts
            time.sleep(poll)
        raise TimeoutError(
            f"Pipeline did not become idle within {timeout}s. Last counts: {last_counts}"
        )

    # -- querying ----------------------------------------------------------

    def query(self, question: str, **params) -> dict:
        """POST /query -> generated answer + references + timing."""
        body = self._build_query_body(question, params)
        resp = self._post("/query", json=body)
        resp.raise_for_status()
        return resp.json()

    def query_data(self, question: str, **params) -> dict:
        """POST /query/data -> structured retrieval (+ citation highlights)."""
        body = self._build_query_body(question, params)
        resp = self._post("/query/data", json=body)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _build_query_body(question: str, params: dict) -> dict:
        body: dict[str, Any] = {"query": question, "mode": config.QUERY_MODE}
        if config.QUERY_TOP_K is not None:
            body["top_k"] = config.QUERY_TOP_K
        if config.QUERY_CHUNK_TOP_K is not None:
            body["chunk_top_k"] = config.QUERY_CHUNK_TOP_K
        if config.QUERY_ENABLE_RERANK is not None:
            body["enable_rerank"] = config.QUERY_ENABLE_RERANK
        if config.INCLUDE_CITATION_HIGHLIGHTS:
            body["include_citation_highlights"] = True
        # Caller overrides (e.g. per-question mode) win.
        body.update({k: v for k, v in params.items() if v is not None})
        return body

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "LightRAGClient":
        return self

    def __exit__(self, *exc) -> None:
        self._client.close()
