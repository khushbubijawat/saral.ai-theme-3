"""Generators for SARAL outputs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

from ..ingestion.chunker import simple_sentence_split
from ..types import AudienceProfile, AudienceStyle, ContentBlock, Duration, GenerationConfig, GenerationOutput, Provenance, RetrievalResult
from .language_models import BaseLanguageModel, DummyLanguageModel
from .prompt_builder import build_prompt
from .safety import describe_safety_rules, enforce_safety

SLIDE_BUDGET = {
    Duration.SHORT_30S: 3,
    Duration.MEDIUM_90S: 5,
    Duration.LONG_5MIN: 7,
}

SCRIPT_SENTENCES = {
    Duration.SHORT_30S: 6,
    Duration.MEDIUM_90S: 14,
    Duration.LONG_5MIN: 30,
}


def _style_prefix(style: AudienceStyle) -> str:
    return {
        AudienceStyle.TECHNICAL: "Technical insight:",
        AudienceStyle.PLAIN: "Plain-language take:",
        AudienceStyle.PRESS: "Headline:",
    }.get(style, "Insight:")


@dataclass
class BaseGenerator:
    def generate_outputs(
        self,
        instruction: str,
        retrievals: List[RetrievalResult],
        config: GenerationConfig,
        audience: AudienceProfile,
    ) -> GenerationOutput:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class RuleBasedGenerator(BaseGenerator):
    def generate_outputs(
        self,
        instruction: str,
        retrievals: List[RetrievalResult],
        config: GenerationConfig,
        audience: AudienceProfile,
    ) -> GenerationOutput:
        sentences: List[ContentBlock] = []
        for result in retrievals:
            split = simple_sentence_split(result.chunk.text)
            for sentence in split:
                safe_sentence = enforce_safety(sentence)
                provenance = [
                    Provenance(chunk_id=result.chunk.chunk_id, page=result.chunk.page, score=result.score)
                ]
                sentences.append(ContentBlock(text=safe_sentence, provenance=provenance))

        script_budget = SCRIPT_SENTENCES[config.duration]
        slides_budget = SLIDE_BUDGET[config.duration]
        slides = []
        notes = []
        tweets = []
        linkedin = []

        for idx in range(min(slides_budget, len(sentences))):
            slides.append(_apply_style(sentences[idx], audience.style, prefix=True, index=idx + 1))
            if config.include_speaker_notes:
                note_text = f"Speaker note {idx + 1}: Expand on {sentences[idx].text} with a visual cue."
                notes.append(ContentBlock(text=note_text, provenance=sentences[idx].provenance))

        script = sentences[:script_budget]

        if config.include_tweets:
            for block in sentences[: slides_budget * 2]:
                tweet_text = f"{block.text} ({block.provenance[0].chunk_id}@p{block.provenance[0].page})"
                tweets.append(ContentBlock(text=tweet_text[:260], provenance=block.provenance))

        if config.include_linkedin:
            chunk = sentences[:5]
            merged = " ".join(b.text for b in chunk)
            linkedin.append(ContentBlock(text=merged[:900], provenance=[p for b in chunk for p in b.provenance]))

        metadata = {
            "instruction": instruction,
            "audience": audience.label,
            "style": audience.style.value,
            "duration": config.duration.value,
        }
        return GenerationOutput(slides=slides, script=script, notes=notes, tweets=tweets, linkedin_summaries=linkedin, metadata=metadata)


def _apply_style(block: ContentBlock, style: AudienceStyle, prefix: bool, index: int) -> ContentBlock:
    if prefix:
        styled = f"Slide {index} - {_style_prefix(style)} {block.text}"
    else:
        styled = f"{_style_prefix(style)} {block.text}"
    return ContentBlock(text=styled, provenance=block.provenance)


@dataclass
class LLMGenerator(BaseGenerator):
    language_model: BaseLanguageModel | None = None

    def __post_init__(self) -> None:
        if self.language_model is None:
            self.language_model = DummyLanguageModel()

    def generate_outputs(
        self,
        instruction: str,
        retrievals: List[RetrievalResult],
        config: GenerationConfig,
        audience: AudienceProfile,
    ) -> GenerationOutput:
        prompt = build_prompt(
            instruction=instruction,
            context_chunks=retrievals,
            audience=audience,
            duration=config.duration,
            safety_directives=describe_safety_rules(),
        )
        format_hint = {
            "slides": ["Slide text", "Provenance chunk id"],
            "script": ["Sentence", "Provenance chunk id"],
            "notes": ["Note", "Reference"],
            "tweets": ["Tweet"],
            "linkedin": ["Summary"],
        }
        prompt += "\nReturn valid JSON with keys: slides, script, notes, tweets, linkedin."
        prompt += f"\nFormat hint: {json.dumps(format_hint)}"
        raw = self.language_model.generate(prompt)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # fallback to deterministic generator
            return RuleBasedGenerator().generate_outputs(instruction, retrievals, config, audience)

        def _blocks(key: str) -> List[ContentBlock]:
            items = data.get(key, [])
            output = []
            for item in items:
                text = item if isinstance(item, str) else item.get("text", "")
                provenance = item.get("provenance") if isinstance(item, dict) else None
                prov_objects = []
                if isinstance(provenance, list):
                    for prov in provenance:
                        prov_objects.append(
                            Provenance(
                                chunk_id=str(prov.get("chunk_id", "?")),
                                page=str(prov.get("page", "?")),
                                score=float(prov.get("score", 0.0)),
                            )
                        )
                output.append(ContentBlock(text=text, provenance=prov_objects))
            return output

        return GenerationOutput(
            slides=_blocks("slides"),
            script=_blocks("script"),
            notes=_blocks("notes"),
            tweets=_blocks("tweets"),
            linkedin_summaries=_blocks("linkedin"),
            metadata={
                "instruction": instruction,
                "audience": audience.label,
                "style": audience.style.value,
                "duration": config.duration.value,
                "generator": "llm",
            },
        )
