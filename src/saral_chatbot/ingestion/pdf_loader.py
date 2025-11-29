"""Utilities to load raw text from PDF, LaTeX, or plain text files."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def load_document(path: str) -> Tuple[str, Dict[int, str]]:
    """Return the doc text and page map keyed by page index."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)

    suffix = file_path.suffix.lower()
    if suffix in {".pdf"}:
        return _load_pdf(file_path)
    if suffix in {".tex", ".txt", ".md"}:
        text = file_path.read_text(encoding="utf-8")
        return text, {idx: page for idx, page in enumerate(text.splitlines(), start=1)}
    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        text = payload.get("text", "")
        pages = payload.get("pages", {})
        return text, {int(k): str(v) for k, v in pages.items()}

    raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(path: Path) -> Tuple[str, Dict[int, str]]:
    if PdfReader is None:
        raise ImportError("pypdf is required to parse PDF files")

    reader = PdfReader(str(path))
    page_text = {}
    for idx, page in enumerate(reader.pages, start=1):
        try:
            content = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to extract page %s: %s", idx, exc)
            content = ""
        page_text[idx] = content
    joined = "\n".join(page_text.values())
    return joined, page_text
