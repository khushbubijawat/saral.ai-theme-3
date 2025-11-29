"""Embedding utilities with pluggable backends."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


class EmbeddingBackend:
    name: str = "base"

    def embed(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class SentenceTransformerBackend(EmbeddingBackend):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # local import

        self.model = SentenceTransformer(self.model_name)
        self.name = f"sentence_transformer::{self.model_name}"

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return self.model.encode(list(texts), convert_to_numpy=True).tolist()


@dataclass
class OpenAIEmbeddingBackend(EmbeddingBackend):
    model: str = "text-embedding-3-small"

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("openai package is required for OpenAI embeddings") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY env var is required")
        self.client = OpenAI(api_key=api_key)
        self.name = f"openai::{self.model}"

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            resp = self.client.embeddings.create(model=self.model, input=text)
            vectors.append(list(resp.data[0].embedding))
        return vectors


class TfidfBackend(EmbeddingBackend):
    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # local import

        self.vectorizer = TfidfVectorizer(max_features=2048)
        self.name = "tfidf"

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        matrix = self.vectorizer.fit_transform(texts)
        return matrix.toarray().tolist()


class EmbeddingManager:
    def __init__(self, backend: EmbeddingBackend | None = None) -> None:
        self.backend = backend or self._default_backend()

    def _default_backend(self) -> EmbeddingBackend:
        try:
            return SentenceTransformerBackend()
        except Exception as exc:  # pragma: no cover - fallback
            LOGGER.warning("Falling back to TF-IDF backend: %s", exc)
            return TfidfBackend()

    def embed(self, texts: Iterable[str]) -> List[np.ndarray]:
        vectors = self.backend.embed(list(texts))
        return [np.array(v, dtype=float) for v in vectors]
