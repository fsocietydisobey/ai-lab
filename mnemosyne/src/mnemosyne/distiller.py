"""Distiller — session transcript → training pairs, no external API.

Extracts meaningful text blocks from Claude Code session JSONL or plain text
transcripts and stores them as instruction-response pairs. The lead session
(which is already Claude) synthesises meaning at query time from the raw pairs.
"""

from __future__ import annotations

import json
import re

from mnemosyne import store


_MIN_CHARS = 100
_MAX_CHARS = 2000

_TOPIC_PATTERNS = [
    (r"key file[s]?[:]\s*(.+?)(?:\n|$)", "What is the key file for {}?"),
    (r"the fix(?:\s+was|:)\s*(.+?)(?:\n|$)", "How was this fixed: {}?"),
    (r"pattern[:]\s*(.+?)(?:\n|$)", "What pattern is used for {}?"),
    (r"footgun[:]\s*(.+?)(?:\n|$)", "What is the footgun around {}?"),
]


def _extract_blocks(transcript: str) -> list[str]:
    blocks: list[str] = []

    # Path 1 — Claude Code JSONL (role=assistant, content=[{type:text}])
    jsonl_hit = False
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        jsonl_hit = True
        role = record.get("role") or record.get("type", "")
        if role != "assistant":
            continue
        content = record.get("content", "")
        if isinstance(content, str) and content.strip():
            blocks.append(content.strip())
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        blocks.append(text)

    if jsonl_hit:
        return blocks

    # Path 2 — plain text with Assistant: / assistant: markers
    for match in re.finditer(
        r"(?:^|\n)(?:Assistant|assistant):\s*(.+?)(?=\n(?:Human|User|assistant|Assistant):|$)",
        transcript,
        re.DOTALL,
    ):
        text = match.group(1).strip()
        if text:
            blocks.append(text)

    if blocks:
        return blocks

    # Path 3 — plain prose: split on double newlines, keep substantial paragraphs
    for para in re.split(r"\n\s*\n", transcript):
        para = para.strip()
        if len(para) >= _MIN_CHARS:
            blocks.append(para)

    return blocks


def _make_instruction(text: str, domain: str) -> str:
    lower = text.lower()
    for pattern, template in _TOPIC_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            topic = m.group(1).strip()[:80]
            return template.format(topic)
    first = re.split(r"[.!?\n]", text)[0].strip()[:100]
    if len(first) > 15:
        return f"What does the {domain} layer do regarding: {first}?"
    return f"What did the {domain} lead learn about {domain} in this session?"


def distill(transcript: str, domain: str, session_slug: str = "") -> list[dict]:
    """Extract and store training pairs from a transcript without any API call."""
    blocks = _extract_blocks(transcript[:50000])
    pairs: list[dict] = []
    for block in blocks:
        if len(block) < _MIN_CHARS:
            continue
        response = block[:_MAX_CHARS]
        instruction = _make_instruction(block, domain)
        store.append(
            domain=domain,
            instruction=instruction,
            response=response,
            source_session=session_slug,
        )
        pairs.append({"instruction": instruction, "response": response})
    return pairs
