"""Text chunking utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

from ..types import RetrievedChunk

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkConfig:
    chunk_size: int = 500
    overlap: int = 120


def simple_sentence_split(text: str) -> List[str]:
    """Lightweight fallback splitter."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(full_text: str, page_map: Dict[int, str], config: ChunkConfig) -> List[RetrievedChunk]:
    sentences = simple_sentence_split(full_text)
    window: List[str] = []
    chunks: List[RetrievedChunk] = []
    pointer = 0
    for sentence in sentences:
        window.append(sentence)
        pointer += len(sentence)
        joined = " ".join(window)
        if len(joined) >= config.chunk_size:
            chunk_id = f"chunk_{len(chunks)}"
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=joined,
                    page=_infer_page(pointer, page_map),
                    embedding=None,
                    metadata={},
                )
            )
            if config.overlap > 0:
                overlap_tokens = " ".join(joined.split()[-config.overlap :])
                window = [overlap_tokens]
            else:
                window = []
    if window:
        chunk_id = f"chunk_{len(chunks)}"
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=" ".join(window),
                page=_infer_page(pointer, page_map),
                embedding=None,
                metadata={},
            )
        )
    return chunks


def _infer_page(pointer: int, page_map: Dict[int, str]) -> str:
    if not page_map:
        return "?"
    cumulative = 0
    for page_number, content in page_map.items():
        cumulative += len(content)
        if pointer <= cumulative:
            return str(page_number)
    return str(max(page_map.keys()))
