"""Prompt construction utilities."""
from __future__ import annotations

from typing import List

from ..types import AudienceProfile, AudienceStyle, Duration, RetrievalResult

STYLE_TIPS = {
    AudienceStyle.TECHNICAL: "Use precise terminology, cite equations inline with LaTeX",
    AudienceStyle.PLAIN: "Favor analogies, avoid acronyms without expansion",
    AudienceStyle.PRESS: "Write punchy, quotable bullets with impact stats",
}

DURATION_BOUNDS = {
    Duration.SHORT_30S: 120,
    Duration.MEDIUM_90S: 260,
    Duration.LONG_5MIN: 700,
}


def build_prompt(
    instruction: str,
    context_chunks: List[RetrievalResult],
    audience: AudienceProfile,
    duration: Duration,
    safety_directives: List[str] | None = None,
) -> str:
    context_text = []
    for idx, result in enumerate(context_chunks, start=1):
        chunk = result.chunk
        context_text.append(
            f"[Source {idx} | chunk={chunk.chunk_id} | page={chunk.page} | score={result.score:.3f}]\n{chunk.text}"
        )
    context_block = "\n\n".join(context_text)
    safety = "\n".join(safety_directives or [])
    tip = STYLE_TIPS.get(audience.style, "")
    max_tokens = DURATION_BOUNDS.get(duration, 300)

    prompt = f"""
You are SARAL, a fact-grounded assistant that writes audience-specific presentations.
Instruction: {instruction}
Audience: {audience.label}
Tone guidance: {', '.join(audience.tone_directives) or 'default'}
Style tips: {tip}
Context budget: {max_tokens} sentences.
Safety requirements: {safety or 'Avoid offensive or exclusionary language. Prefer accessible descriptions.'}
Use inline citations of the form (chunk-id @ page) for every sentence referencing the sources below.

Sources:\n{context_block}
"""
    return prompt.strip()
