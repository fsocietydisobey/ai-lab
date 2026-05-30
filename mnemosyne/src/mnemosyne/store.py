"""Training data store — append-only JSONL per domain."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent.parent / "data"


def domain_path(domain: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{domain}.jsonl"


def append(domain: str, instruction: str, response: str, source_session: str = "") -> str:
    """Append a training pair. Returns the new record's id."""
    record_id = str(uuid.uuid4())
    record = {
        "id": record_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "source_session": source_session,
        "instruction": instruction,
        "response": response,
        "superseded_by": None,
    }
    with domain_path(domain).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record_id


def _load_raw(domain: str) -> list[dict]:
    """Return ALL records including tombstones, unfiltered."""
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


def load(domain: str) -> list[dict]:
    """Return data records with superseded pairs filtered out."""
    all_records = _load_raw(domain)
    superseded_ids = {
        r["target_id"]
        for r in all_records
        if r.get("type") == "supersede"
    }
    return [
        r for r in all_records
        if r.get("type") != "supersede" and r.get("id") not in superseded_ids
    ]


def supersede(domain: str, target_id: str, by_id: str, reason: str = "") -> None:
    """Mark target_id as superseded by by_id via an appended tombstone record.

    Raises ValueError if target_id is not found or is already superseded.
    """
    all_records = _load_raw(domain)
    data_records = [r for r in all_records if r.get("type") != "supersede"]
    if not any(r.get("id") == target_id for r in data_records):
        raise ValueError(f"record id={target_id!r} not found in domain {domain!r}")
    already_superseded = {
        r["target_id"] for r in all_records if r.get("type") == "supersede"
    }
    if target_id in already_superseded:
        raise ValueError(f"record id={target_id!r} is already superseded in domain {domain!r}")
    tombstone = {
        "type": "supersede",
        "ts": datetime.now(timezone.utc).isoformat(),
        "target_id": target_id,
        "by_id": by_id,
        "reason": reason,
    }
    with domain_path(domain).open("a", encoding="utf-8") as f:
        f.write(json.dumps(tombstone) + "\n")


def domains() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return [p.stem for p in DATA_DIR.glob("*.jsonl")]
