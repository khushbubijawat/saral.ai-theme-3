# Demo walkthrough

Command:
```
python scripts/demo_chat.py --paper data/sample_papers/green_hydrogen.txt \
    --instruction "Make a 7-slide talk for policymakers focusing on finance levers; include notes." \
    --audience policymakers --style plain --duration 90s --save-log conversation_logs/demo_session.json
```

Highlights:
- Bot ingests the paper and returns slide bullets, script snippets, notes, tweets, and LinkedIn summary with provenance (chunk id + page) per sentence.
- Revision command `revise slides 1 make it more visual and shorter` rewrites slide 1, appending a visual cue and truncating text. The change log captures before/after text and rationale.
- Conversation JSON is stored at `conversation_logs/demo_session.json` for audit.
