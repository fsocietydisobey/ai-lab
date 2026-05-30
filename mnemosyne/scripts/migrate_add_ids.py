"""One-off migration: backfill id + superseded_by fields to existing mnemosyne records.

Usage (from ai-lab/mnemosyne root):
    .venv/bin/python3 scripts/migrate_add_ids.py [--dry-run]

Idempotent: records that already have an 'id' field are skipped unchanged.
Tombstone records (type == "supersede") are left untouched.
Backs up each file to <name>.jsonl.bak before rewriting.
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def migrate(dry_run: bool = False) -> None:
    if not DATA_DIR.exists():
        print(f"Data dir not found: {DATA_DIR}")
        sys.exit(1)

    jsonl_files = sorted(DATA_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found — nothing to migrate.")
        return

    for src in jsonl_files:
        records = []
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        needs_backfill = [
            r for r in records
            if r.get("type") != "supersede" and "id" not in r
        ]

        if not needs_backfill:
            print(f"[{src.name}] {len(records)} record(s) — all already migrated, skipping.")
            continue

        print(
            f"[{src.name}] {len(needs_backfill)} record(s) need id backfill "
            f"({len(records) - len(needs_backfill)} already have id or are tombstones)."
        )

        if dry_run:
            print(f"[{src.name}] DRY RUN — no changes written.")
            continue

        bak = src.with_suffix(".jsonl.bak")
        shutil.copy2(src, bak)
        print(f"[{src.name}] backed up to {bak.name}")

        updated = []
        backfilled = 0
        for r in records:
            if r.get("type") == "supersede" or "id" in r:
                updated.append(r)
            else:
                r = dict(r)
                r["id"] = str(uuid.uuid4())
                r["superseded_by"] = None
                updated.append(r)
                backfilled += 1

        src.write_text(
            "\n".join(json.dumps(rec) for rec in updated) + "\n",
            encoding="utf-8",
        )
        print(f"[{src.name}] backfilled {backfilled} record(s).")

        check = []
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    check.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        assert len(check) == len(records), (
            f"Record count mismatch after migration: {len(check)} != {len(records)}"
        )
        missing_ids = [
            r for r in check
            if r.get("type") != "supersede" and "id" not in r
        ]
        assert not missing_ids, f"Records still missing id after migration: {missing_ids}"
        print(f"[{src.name}] verified: {len(check)} record(s), all data records have id.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    migrate(dry_run=dry_run)
