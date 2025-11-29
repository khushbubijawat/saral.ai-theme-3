"""Common dataclasses and enums for the SARAL chatbot pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Sequence


class AudienceStyle(str, Enum):
    TECHNICAL = "technical"
    PLAIN = "plain"
    PRESS = "press"


class Duration(str, Enum):
    SHORT_30S = "30s"
    MEDIUM_90S = "90s"
    LONG_5MIN = "5min"


@dataclass
class AudienceProfile:
    label: str
    style: AudienceStyle
    expertise_notes: str = ""
    tone_directives: Sequence[str] = field(default_factory=list)


@dataclass
class GenerationConfig:
    duration: Duration
    style: AudienceStyle
    include_tweets: bool = True
    include_linkedin: bool = True
    include_speaker_notes: bool = True
    safety_checks: bool = True


@dataclass
class Provenance:
    chunk_id: str
    page: Optional[str]
    score: float


@dataclass
class ContentBlock:
    text: str
    provenance: List[Provenance] = field(default_factory=list)


@dataclass
class GenerationOutput:
    slides: List[ContentBlock]
    script: List[ContentBlock]
    notes: List[ContentBlock]
    tweets: List[ContentBlock]
    linkedin_summaries: List[ContentBlock]
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ChangeRecord:
    timestamp: datetime
    user_request: str
    target_section: str
    before: str
    after: str
    rationale: str


@dataclass
class ConversationTurn:
    role: str
    content: str
    output_snapshot: Optional[GenerationOutput] = None
    change_record: Optional[ChangeRecord] = None


@dataclass
class ConversationLog:
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "change_record": (
                        {
                            "timestamp": t.change_record.timestamp.isoformat()
                            if t.change_record
                            else None,
                            "user_request": t.change_record.user_request if t.change_record else None,
                            "target_section": t.change_record.target_section if t.change_record else None,
                            "before": t.change_record.before if t.change_record else None,
                            "after": t.change_record.after if t.change_record else None,
                            "rationale": t.change_record.rationale if t.change_record else None,
                        }
                        if t.change_record
                        else None
                    ),
                }
                for t in self.turns
            ],
        }


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    page: Optional[str]
    embedding: Optional[List[float]]
    metadata: Dict[str, str]


@dataclass
class RetrievalResult:
    chunk: RetrievedChunk
    score: float


@dataclass
class EvaluationRecord:
    paper_id: str
    audience: str
    rouge_l: float
    semantic_similarity: float
    provenance_coverage: float
    citation_coverage: float
