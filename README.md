# SARAL Theme 3 - Audience-Adaptive RAG Chatbot

This repository contains a prototype SARAL chatbot that turns long-form research inputs (PDF/paper text) into audience-aware slide bullets, scripts, speaker notes, tweet threads, and provenance traces. The system is centred around a retrieval augmented generation (RAG) pipeline with modular ingestion, embedding, retrieval, generation, change tracking, and evaluation utilities.

## Key capabilities

- Paper ingestion: load PDF/LaTeX/text, normalise, and chunk with overlap to preserve context.
- Hybrid embeddings: dense sentence-transformers (or OpenAI embeddings) with TF-IDF fallback for fully offline runs.
- Retrieval + provenance: store chunk metadata, surface references (chunk id, page) next to each generated sentence.
- Audience-aware generation: configurable outputs (30s / 90s / 5min) and styles (technical, plain-English, press) with enforcement of accessibility and safety filters.
- Iterative editing: chat session keeps version history. When a user asks for a revision (e.g., "make #2 less technical"), the delta is computed and reported automatically.
- Minimal chat UI: CLI script demonstrates multi-turn interactions, including provenance display and change tracking.
- Evaluation harness: compute provenance coverage, ROUGE-L, and a light-weight semantic similarity score against human-authored baselines for three sample papers.

## Project layout

```
- conversation_logs/      # Captured chat transcripts for evidence of iteration
- data/sample_papers/     # Small demo corpora (text extractions from PDFs)
- examples/conversations/ # Markdown example walkthroughs
- examples/evaluations/   # Evaluation configs + reference scripts
- scripts/                # CLI entry-points (demo chat, ingestion, evaluation)
- src/saral_chatbot/
  - ingestion/            # PDF/text loaders and chunker
  - embeddings/           # Embedding manager with pluggable backends
  - retrieval/            # Vector store + similarity search
  - generation/           # Prompt builder + generator wrappers
  - ui/                   # Chat flow helpers
  - evaluation/           # Metrics + provenance coverage utilities
  - pipeline.py           # High-level SARALChatbot orchestrator
```

## Getting started

1. Environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Optional: set OPENAI_API_KEY or HF_TOKEN to switch to hosted LLMs/embeddings.

2. Ingest a paper
   ```bash
   python scripts/demo_chat.py --paper data/sample_papers/green_hydrogen.txt \
       --audience policymakers --style plain --duration 90s
   ```
   - To try a fully open-source generator (free), add `--generator hf --hf-model google/flan-t5-small` (requires transformers + torch and a cached/local model).

3. Run evaluation
   ```bash
   python scripts/run_evaluation.py --config examples/evaluations/eval_config.yaml
   ```

## Evaluation summary

examples/evaluations/test_run_report.json shows automatic metrics on the bundled trio of papers. Each generated script cites retrieved chunks, and citation coverage plus ROUGE-L are reported per audience style.

## Notes

- The generator defaults to a RuleBasedGenerator for deterministic demos, but you can enable HFTextGenerationModel or other HuggingFace text-generation models via config.
- Safety and accessibility filters apply lexical checks; extend generation/safety.py to integrate policy engines.
- Conversation histories (JSON + Markdown) inside conversation_logs/ demonstrate iterative edits, change reasoning, and provenance display.

## Next steps

- Swap in production-grade LLM APIs, add guardrails, and wire into SARAL's orchestration layer.
- Extend evaluation to include human annotation pipelines (template in examples/evaluations/human_eval_template.md).
- Containerise (Dockerfile) and deploy behind SARAL chat frontend once API keys and infra are provisioned.
