"""Evaluation helpers for SARAL outputs."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from rouge_score import rouge_scorer

from ..embeddings.embedding_manager import EmbeddingManager
from ..types import ContentBlock, EvaluationRecord, GenerationOutput

LOGGER = logging.getLogger(__name__)


@dataclass
class SimilarityComputer:
    embedding_manager: EmbeddingManager | None = None

    def __post_init__(self) -> None:
        self.embedding_manager = self.embedding_manager or EmbeddingManager()

    def score(self, generated: str, reference: str) -> float:
        vectors = self.embedding_manager.embed([generated, reference])
        a, b = vectors
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
        return float(np.dot(a, b) / denom)


def rouge_l_score(generated: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return float(scores["rougeL"].fmeasure)


def provenance_coverage(blocks: List[ContentBlock]) -> float:
    if not blocks:
        return 0.0
    total = len(blocks)
    cited = sum(1 for block in blocks if block.provenance)
    return cited / total


def citation_coverage(output: GenerationOutput) -> float:
    blocks = output.slides + output.script + output.notes
    return provenance_coverage(blocks)


def evaluate_output(
    output: GenerationOutput,
    reference_script: str,
    paper_id: str,
    audience: str,
    similarity: SimilarityComputer | None = None,
) -> EvaluationRecord:
    similarity = similarity or SimilarityComputer()
    generated_text = " ".join(block.text for block in output.script)
    rouge = rouge_l_score(generated_text, reference_script)
    sim = similarity.score(generated_text, reference_script)
    prov = provenance_coverage(output.script)
    cite = citation_coverage(output)
    return EvaluationRecord(
        paper_id=paper_id,
        audience=audience,
        rouge_l=rouge,
        semantic_similarity=sim,
        provenance_coverage=prov,
        citation_coverage=cite,
    )
