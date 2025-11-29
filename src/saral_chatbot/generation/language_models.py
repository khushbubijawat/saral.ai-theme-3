"""Lightweight wrappers around language model providers."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


class BaseLanguageModel:
    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class OpenAILanguageModel(BaseLanguageModel):
    model: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install openai to use OpenAILanguageModel") from exc
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY env variable missing")
        self.client = OpenAI()

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
        )
        return response.output[0].content[0].text


class DummyLanguageModel(BaseLanguageModel):
    """A deterministic stub that simply returns the prompt."""

    def generate(self, prompt: str) -> str:
        payload = {"prompt_echo": prompt[-400:]}
        return json.dumps(payload)


@dataclass
class HFTextGenerationModel(BaseLanguageModel):
    """
    Lightweight open-source text generator using Hugging Face transformers.
    Defaults to a small model to keep resource use modest.
    """

    model_name: str = "google/flan-t5-small"
    device: str = "cpu"
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install transformers to use HFTextGenerationModel") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        if self.device != "cpu":
            self.model.to(self.device)

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
