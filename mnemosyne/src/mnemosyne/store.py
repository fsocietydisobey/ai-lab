"""Training data store — append-only JSONL per domain."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent.parent / "data"


def domain_path(domain: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{domain}.jsonl"


def append(domain: str, instruction: str, response: str, source_session: str = "") -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "source_session": source_session,
        "instruction": instruction,
        "response": response,
    }
    with domain_path(domain).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load(domain: str) -> list[dict]:
    path = domain_path(domain)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def domains() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return [p.stem for p in DATA_DIR.glob("*.jsonl")]
