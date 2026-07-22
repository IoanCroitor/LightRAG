"""PDF text extraction for ground-truth generation.

Uses ``pypdf`` (already a transitive dependency of LightRAG). The extractor is
defensive: malformed PDF objects emit warnings but never abort, and we cap the
returned text so the ground-truth LLM prompt stays within context limits.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pypdf

logger = logging.getLogger("parsebench.pdf")

# pypdf is noisy about malformed PDFs in this benchmark corpus; silence it.
warnings.filterwarnings("ignore", module="pypdf")


def extract_pdf_text(pdf_path: Path, max_chars: int) -> tuple[str, int]:
    """Return ``(text, page_count)`` for ``pdf_path``.

    Text is truncated to ``max_chars`` characters (from the start) to keep the
    ground-truth prompt bounded. Multi-column layouts interleave, which is
    acceptable: the LLM is asked for facts that survive the layout.
    """
    pdf_path = Path(pdf_path)
    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to open %s: %s", pdf_path.name, exc)
        return "", 0

    page_count = len(reader.pages)
    parts: list[str] = []
    total = 0
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Page extract error in %s: %s", pdf_path.name, exc)
            page_text = ""
        # Normalize whitespace lightly without dropping structure.
        page_text = " ".join(page_text.split())
        parts.append(page_text)
        total += len(page_text)
        if total >= max_chars:
            break

    text = " ".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text, page_count
