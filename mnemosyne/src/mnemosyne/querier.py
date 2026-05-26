"""Querier — answer domain memory questions from lead sessions.

Phase 1: uses the base model + injected training data as context (no fine-tune yet).
Phase 2: queries the LoRA-adapted model directly once training pipeline is ready.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic

from mnemosyne import store


_CLIENT = anthropic.Anthropic()
_TOP_K = 20  # max training pairs to inject per query


def query(domain: str, question: str) -> str:
    """Answer a domain memory question using accumulated training data as context."""
    pairs = store.load(domain)
    if not pairs:
        return f"No accumulated memory for domain '{domain}' yet."

    # Most recent pairs first (recency bias — newer knowledge takes precedence)
    pairs = pairs[-_TOP_K:]

    context_blocks = "\n\n".join(
        f"Q: {p['instruction']}\nA: {p['response']}" for p in pairs
    )

    system = f"""\
You are a domain memory oracle for the '{domain}' engineering domain.
Answer questions using ONLY the accumulated knowledge below — do not add
general best practices or assumptions. If the knowledge doesn't cover the
question, say so explicitly.

Accumulated domain knowledge:
{context_blocks}
"""
    msg = _CLIENT.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return msg.content[0].text.strip()
