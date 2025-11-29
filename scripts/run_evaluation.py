"""Run automatic evaluation for sample papers."""
from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from saral_chatbot.evaluation.metrics import evaluate_output
from saral_chatbot.pipeline import SaralChatbot
from saral_chatbot.types import AudienceProfile, AudienceStyle, Duration, GenerationConfig

app = typer.Typer(help="Automatic evaluation harness")


@app.command()
def run(
    config: Path = typer.Option(..., help="YAML config describing evaluation set"),
    output: Path = typer.Option(Path("examples/evaluations/test_run_report.json"), help="Where to store metrics"),
):
    plan = yaml.safe_load(config.read_text())
    records = []
    for item in plan.get("cases", []):
        typer.echo(f"Evaluating {item['paper']} for {item['audience']}")
        bot = SaralChatbot()
        bot.ingest(item["paper"])
        profile = AudienceProfile(
            label=item["audience"],
            style=AudienceStyle(item["style"]),
            tone_directives=item.get("tone", []),
        )
        config_obj = GenerationConfig(
            duration=Duration(item["duration"]),
            style=profile.style,
        )
        output_bundle = bot.generate(item["instruction"], profile, config_obj)
        reference_text = Path(item["reference"]).read_text(encoding="utf-8")
        record = evaluate_output(
            output_bundle,
            reference_text,
            paper_id=item.get("paper_id", Path(item["paper"]).stem),
            audience=profile.label,
        )
        records.append(record.__dict__)
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    typer.echo(f"Saved results to {output}")


if __name__ == "__main__":
    app()
