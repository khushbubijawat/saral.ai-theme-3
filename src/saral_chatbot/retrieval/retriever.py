"""Retriever orchestrating embeddings + vector store."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from ..embeddings.embedding_manager import EmbeddingManager
from ..types import RetrievalResult, RetrievedChunk
from .vector_store import VectorStore


@dataclass
class Retriever:
    embedding_manager: EmbeddingManager
    store: VectorStore

    @classmethod
    def from_chunks(cls, chunks: Iterable[RetrievedChunk], embedding_manager: EmbeddingManager | None = None) -> "Retriever":
        manager = embedding_manager or EmbeddingManager()
        store = VectorStore()
        texts = [chunk.text for chunk in chunks]
        embeddings = manager.embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            store.add(chunk, emb)
            chunk.embedding = emb.tolist()
        return cls(embedding_manager=manager, store=store)

    def query(self, prompt: str, k: int = 5) -> List[RetrievalResult]:
        query_vector = self.embedding_manager.embed([prompt])[0]
        return self.store.similarity_search(query_vector, k=k)

    def to_serializable(self) -> List[dict]:
        payload = []
        for chunk in self.store.chunks:
            payload.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "page": chunk.page,
                    "metadata": chunk.metadata,
                }
            )
        return payload
