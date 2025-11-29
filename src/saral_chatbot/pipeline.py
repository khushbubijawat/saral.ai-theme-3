"""High-level SARAL chatbot orchestrator."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

from .embeddings.embedding_manager import EmbeddingManager
from .generation.generator import LLMGenerator, RuleBasedGenerator
from .generation.language_models import BaseLanguageModel
from .generation.safety import describe_safety_rules
from .ingestion.chunker import ChunkConfig, chunk_text
from .ingestion.pdf_loader import load_document
from .retrieval.retriever import Retriever
from .types import (
    AudienceProfile,
    AudienceStyle,
    ChangeRecord,
    ConversationLog,
    ConversationTurn,
    Duration,
    GenerationConfig,
    GenerationOutput,
    RetrievalResult,
)


@dataclass
class SaralChatbot:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding_manager: EmbeddingManager | None = None
    language_model: BaseLanguageModel | None = None
    use_llm: bool = False

    def __post_init__(self) -> None:
        self.embedding_manager = self.embedding_manager or EmbeddingManager()
        self.conversation = ConversationLog(session_id=self.session_id)
        self.generator = (
            LLMGenerator(language_model=self.language_model)
            if self.use_llm
            else RuleBasedGenerator()
        )
        self.retriever: Retriever | None = None
        self.sources_path: Path | None = None
        self.current_output: GenerationOutput | None = None

    # --------------------------- INGESTION ---------------------------
    def ingest(self, path: str, chunk_config: ChunkConfig | None = None) -> None:
        text, page_map = load_document(path)
        chunk_config = chunk_config or ChunkConfig()
        chunks = chunk_text(text, page_map, chunk_config)
        self.retriever = Retriever.from_chunks(chunks, self.embedding_manager)
        self.sources_path = Path(path)

    # --------------------------- GENERATION ---------------------------
    def generate(
        self,
        request: str,
        audience: AudienceProfile,
        config: GenerationConfig,
        top_k: int = 5,
    ) -> GenerationOutput:
        if self.retriever is None:
            raise RuntimeError("Call ingest() before generate().")
        retrievals = self.retriever.query(request, k=top_k)
        output = self.generator.generate_outputs(request, retrievals, config, audience)
        self.current_output = output
        self._log_turn("user", request)
        self._log_turn("assistant", _summarise_output(output), output)
        return output

    # --------------------------- CHANGE TRACKING ---------------------------
    def revise_section(self, section: str, index: int, directive: str) -> ChangeRecord:
        if self.current_output is None:
            raise RuntimeError("No generation available to revise.")
        section_map = {
            "slides": self.current_output.slides,
            "script": self.current_output.script,
            "notes": self.current_output.notes,
        }
        if section not in section_map:
            raise ValueError(f"Section {section} not supported for revisions")
        blocks = section_map[section]
        if not (0 <= index < len(blocks)):
            raise IndexError("Index outside section range")
        before = blocks[index].text
        after = _apply_directive(before, directive)
        blocks[index].text = after
        change = ChangeRecord(
            timestamp=datetime.utcnow(),
            user_request=directive,
            target_section=f"{section}[{index}]",
            before=before,
            after=after,
            rationale=f"Applied directive '{directive}' via rule-based mutation.",
        )
        self._log_turn("user", directive)
        self._log_turn("assistant", f"Updated {section}[{index}]", change_record=change)
        return change

    # --------------------------- CONVERSATION LOGGING ---------------------------
    def save_conversation(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.conversation.to_dict(), indent=2), encoding="utf-8")

    def _log_turn(
        self,
        role: str,
        content: str,
        output: GenerationOutput | None = None,
        change_record: ChangeRecord | None = None,
    ) -> None:
        self.conversation.turns.append(
            ConversationTurn(role=role, content=content, output_snapshot=output, change_record=change_record)
        )


# ---------------------------------------------------------------------------


def _apply_directive(text: str, directive: str) -> str:
    directive_lower = directive.lower()
    if "less technical" in directive_lower:
        replacements = {"electrolyzer": "hydrogen machine", "provenance": "source"}
        for old, new in replacements.items():
            text = text.replace(old, new)
    if "more visual" in directive_lower:
        text += " [Add: photo cue or chart icon]"
    if "shorter" in directive_lower:
        tokens = text.split()
        text = " ".join(tokens[: max(5, len(tokens) // 2)]) + "..."
    return text


def _summarise_output(output: GenerationOutput) -> str:
    return (
        f"Slides={len(output.slides)}, script sentences={len(output.script)}, "
        f"notes={len(output.notes)}, tweets={len(output.tweets)}"
    )
