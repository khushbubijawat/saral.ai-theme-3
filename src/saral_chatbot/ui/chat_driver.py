"""Command-line chat helper."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from ..pipeline import SaralChatbot
from ..types import AudienceProfile, AudienceStyle, Duration, GenerationConfig

console = Console()


def _audience_profile(label: str, style: str) -> AudienceProfile:
    try:
        style_enum = AudienceStyle(style)
    except ValueError:
        style_enum = AudienceStyle.PLAIN
    tone = ["accessible", "cite sources"] if style_enum != AudienceStyle.TECHNICAL else ["precise", "math ok"]
    return AudienceProfile(label=label.title(), style=style_enum, tone_directives=tone)


@dataclass
class ChatDriver:
    bot: SaralChatbot

    def run(
        self,
        paper: str,
        instruction: str,
        audience_label: str,
        style: str,
        duration: str,
        save_log: Optional[str] = None,
    ) -> None:
        console.log(f"Ingesting {paper}")
        self.bot.ingest(paper)
        profile = _audience_profile(audience_label, style)
        config = GenerationConfig(duration=Duration(duration), style=profile.style)
        output = self.bot.generate(instruction, profile, config)
        _render_output(output)
        console.print("Type a revision command (e.g., revise slides 1 make it more visual) or 'exit'.")
        while True:
            user = console.input("[bold green]You> [/]")
            if user.strip().lower() in {"exit", "quit"}:
                break
            parsed = _parse_revision(user)
            if not parsed:
                console.print("Could not parse command. Format: revise <section> <index> <directive>")
                continue
            section, index, directive = parsed
            record = self.bot.revise_section(section, index, directive)
            console.print(f"Applied change: {record.target_section}")
            console.print(f"Before: {record.before}\nAfter: {record.after}")
        if save_log:
            self.bot.save_conversation(save_log)
            console.print(f"Conversation saved to {save_log}")


def _parse_revision(text: str):
    parts = text.strip().split()
    if len(parts) < 4 or parts[0].lower() != "revise":
        return None
    section = parts[1].lower()
    try:
        index = int(parts[2]) - 1
    except ValueError:
        return None
    directive = " ".join(parts[3:])
    return section, index, directive


def _render_output(output):
    table = Table(title="Slide bullets")
    table.add_column("#")
    table.add_column("Text")
    table.add_column("Provenance")
    for idx, block in enumerate(output.slides, start=1):
        prov = ", ".join(f"{p.chunk_id}@p{p.page}" for p in block.provenance)
        table.add_row(str(idx), block.text, prov)
    console.print(table)
    console.print("\n[bold]Script (first 5 sentences)[/]")
    for block in output.script[:5]:
        prov = ", ".join(f"{p.chunk_id}@p{p.page}" for p in block.provenance)
        console.print(f"- {block.text} [{prov}]")

    console.print("\n[bold]Tweets[/]")
    for tweet in output.tweets[:3]:
        console.print(f"- {tweet.text}")

    console.print("\n[bold]LinkedIn summary[/]")
    if output.linkedin_summaries:
        console.print(output.linkedin_summaries[0].text)
