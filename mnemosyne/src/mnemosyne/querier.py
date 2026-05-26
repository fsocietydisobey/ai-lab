"""Querier — answer domain memory questions using accumulated training pairs as context."""

from __future__ import annotations

import anthropic

from mnemosyne import store


_TOP_K = 20


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def query(domain: str, question: str) -> str:
    pairs = store.load(domain)
    if not pairs:
        return f"No accumulated memory for domain '{domain}' yet."

    context_blocks = "\n\n".join(
        f"Q: {p['instruction']}\nA: {p['response']}" for p in pairs[-_TOP_K:]
    )
    system = (
        f"You are a domain memory oracle for the '{domain}' engineering domain. "
        f"Answer questions using ONLY the accumulated knowledge below — do not add "
        f"general best practices or assumptions. If the knowledge doesn't cover the "
        f"question, say so explicitly.\n\nAccumulated domain knowledge:\n{context_blocks}"
    )
    msg = _client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return msg.content[0].text.strip()
