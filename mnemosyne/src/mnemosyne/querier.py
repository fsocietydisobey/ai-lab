"""Querier — return accumulated training pairs for a domain, no external API.

The lead session (which is already Claude) reads the returned pairs and
synthesises meaning itself. No Haiku call needed.
"""

from __future__ import annotations

from mnemosyne import store


_TOP_K = 20


def query(domain: str, question: str) -> str:
    """Return top-K pairs for a domain as a formatted context block.

    The caller (a domain lead session) synthesises the answer using its own
    Claude intelligence — no extra API call required.
    """
    pairs = store.load(domain)
    if not pairs:
        return f"No accumulated memory for domain '{domain}' yet."

    recent = pairs[-_TOP_K:]
    lines = [f"Q: {p['instruction']}\nA: {p['response']}" for p in recent]
    header = f"# Domain memory — {domain} ({len(recent)} of {len(pairs)} pairs)\n"
    return header + "\n\n".join(lines)
