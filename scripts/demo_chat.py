"""Demo CLI for SARAL chatbot."""
from __future__ import annotations

from typing import Optional

import typer

from saral_chatbot.generation.language_models import HFTextGenerationModel
from saral_chatbot.pipeline import SaralChatbot
from saral_chatbot.types import AudienceStyle, Duration
from saral_chatbot.ui.chat_driver import ChatDriver

app = typer.Typer(help="Interactive SARAL chatbot demo")


def _style(value: str) -> str:
    try:
        AudienceStyle(value)
    except ValueError as exc:
        raise typer.BadParameter("Choose from technical/plain/press") from exc
    return value


def _duration(value: str) -> str:
    try:
        Duration(value)
    except ValueError as exc:
        raise typer.BadParameter("Choose from 30s/90s/5min") from exc
    return value


@app.command()
def chat(
    paper: str = typer.Option(..., help="Path to PDF/text input"),
    instruction: str = typer.Option(..., help="User instruction for the bot"),
    audience: str = typer.Option("policymakers", help="Audience label"),
    style: str = typer.Option("plain", callback=_style),
    duration: str = typer.Option("90s", callback=_duration),
    generator: str = typer.Option("rule", help="rule or hf"),
    hf_model: str = typer.Option(
        "google/flan-t5-small", help="HF model name/path (used when generator=hf)"
    ),
    save_log: Optional[str] = typer.Option(None, help="Where to store the conversation JSON"),
):
    if generator == "hf":
        lm = HFTextGenerationModel(model_name=hf_model)
        bot = SaralChatbot(language_model=lm, use_llm=True)
    else:
        bot = SaralChatbot()
    driver = ChatDriver(bot)
    driver.run(
        paper=paper,
        instruction=instruction,
        audience_label=audience,
        style=style,
        duration=duration,
        save_log=save_log,
    )


if __name__ == "__main__":
    app()
