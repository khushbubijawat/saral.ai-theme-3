"""Lightweight lexical safety and accessibility checks."""
from __future__ import annotations

import re
from typing import List

BLOCKED_TERMS = {
    "hate",
    "kill",
    "slur",
}

JARGON_MAP = {
    "non-stationary diffusion": "changing diffusion patterns",
    "anthropogenic": "human-caused",
}


def enforce_safety(text: str) -> str:
    lowered = text.lower()
    for term in BLOCKED_TERMS:
        if term in lowered:
            text = text.replace(term, "[removed]")
    for jargon, simple in JARGON_MAP.items():
        text = re.sub(jargon, f"{simple} (formerly \"{jargon}\")", text, flags=re.IGNORECASE)
    return text


def describe_safety_rules() -> List[str]:
    return [
        "Do not include hateful or violent language.",
        "Prefer people-first, accessible phrasing.",
        "Flag if provenance is missing before presenting a claim.",
    ]
