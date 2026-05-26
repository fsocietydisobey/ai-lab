"""Distiller — extract structured training pairs from a session transcript.

Calls Claude API to distill a raw session transcript into instruction-response
pairs suitable for LoRA fine-tuning. Each pair captures one domain insight:
a pattern, footgun, key file note, or open question the lead discovered.
"""

from __future__ import annotations

import json
import anthropic

from mnemosyne import store


_CLIENT = anthropic.Anthropic()

_SYSTEM = """\
You are a knowledge distiller for a software engineering AI system. You receive
a session transcript from a domain lead (a specialist AI agent) and extract
structured knowledge that should persist across session boundaries.

Output a JSON array of instruction-response pairs. Each pair captures ONE
distinct insight the lead gained — a pattern, footgun, key file, design
decision, or bug fix. Aim for 5-20 pairs per transcript.

Format each pair as:
{
  "instruction": "What does this codebase do regarding <specific topic>?",
  "response": "<concise, specific answer based on what the lead actually learned>"
}

Rules:
- Only extract knowledge that generalizes beyond this specific task
- Instruction must be a natural question a future lead would ask
- Response must be specific to THIS codebase, not general best practices
- Skip tool call noise, errors, and intermediate reasoning
- Prefer concrete over abstract ("uses controlled inputs via X pattern" > "uses React patterns")
"""


def distill(transcript: str, domain: str, session_slug: str = "") -> list[dict]:
    """Distill a session transcript into training pairs and append to store."""
    msg = _CLIENT.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system=_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Domain: {domain}\n\nTranscript:\n{transcript[:50000]}",
            }
        ],
    )
    text = msg.content[0].text.strip()
    # Extract JSON array from response
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    pairs = json.loads(text[start:end])
    for pair in pairs:
        store.append(
            domain=domain,
            instruction=pair.get("instruction", ""),
            response=pair.get("response", ""),
            source_session=session_slug,
        )
    return pairs
