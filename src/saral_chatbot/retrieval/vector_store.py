"""Simple in-memory vector store."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..types import RetrievedChunk, RetrievalResult


@dataclass
class VectorStore:
    embeddings: List[np.ndarray] = field(default_factory=list)
    chunks: List[RetrievedChunk] = field(default_factory=list)

    def add(self, chunk: RetrievedChunk, embedding: np.ndarray) -> None:
        self.chunks.append(chunk)
        self.embeddings.append(embedding)

    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[RetrievalResult]:
        if not self.embeddings:
            return []
        matrix = np.vstack(self.embeddings)
        # cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        scores = doc_norms @ query_norm
        best_idx = scores.argsort()[-k:][::-1]
        return [RetrievalResult(chunk=self.chunks[i], score=float(scores[i])) for i in best_idx]
