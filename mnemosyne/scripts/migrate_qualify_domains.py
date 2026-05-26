"""One-off migration: qualify mnemosyne domain keys to <project>:<domain>.

Usage (from ai-lab/mnemosyne root):
    .venv/bin/python3 scripts/migrate_qualify_domains.py [--dry-run]

Rewrites existing bare-domain JSONL files to qualified keys:
    backend  → khimaira:backend
    data     → khimaira:data
    devops   → khimaira:devops

Idempotent: entries with already-qualified domains (containing ':') are skipped.
Backs up each file to <name>.jsonl.bak before rewriting.

Only rewrites the known khimaira domains — does not touch any other JSONL files.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Mapping of bare domain → qualified domain for the khimaira project
QUALIFY_MAP: dict[str, str] = {
    "backend": "khimaira:backend",
    "data": "khimaira:data",
    "devops": "khimaira:devops",
}


def migrate(dry_run: bool = False) -> None:
    if not DATA_DIR.exists():
        print(f"Data dir not found: {DATA_DIR}")
        sys.exit(1)

    for bare_domain, qualified in QUALIFY_MAP.items():
        src = DATA_DIR / f"{bare_domain}.jsonl"
        if not src.exists():
            print(f"[{bare_domain}] no file — skipping")
            continue

        records = []
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        needs_migration = [r for r in records if ":" not in r.get("domain", "")]
        already_qualified = [r for r in records if ":" in r.get("domain", "")]

        if not needs_migration:
            print(f"[{bare_domain}] all {len(already_qualified)} entries already qualified — skipping")
            continue

        print(f"[{bare_domain}] {len(needs_migration)} entries to migrate → {qualified} "
              f"(+ {len(already_qualified)} already qualified)")

        if dry_run:
            print(f"[{bare_domain}] DRY RUN — no changes written")
            continue

        # Backup before rewriting
        bak = src.with_suffix(".jsonl.bak")
        shutil.copy2(src, bak)
        print(f"[{bare_domain}] backed up to {bak}")

        # Rewrite: update domain field for unqualified entries
        dst = DATA_DIR / f"{qualified.replace(':', '__')}.jsonl.tmp"
        qualified_records = []
        for r in records:
            if ":" not in r.get("domain", ""):
                r = dict(r)
                r["domain"] = qualified
            qualified_records.append(r)

        # Write to new qualified file + remove old bare file
        new_path = DATA_DIR / f"{qualified}.jsonl".replace(":", ":")  # keep ':' in filename
        # Filesystems allow ':' on Linux; use it directly
        new_path = DATA_DIR / f"{qualified}.jsonl"
        with new_path.open("w", encoding="utf-8") as f:
            for r in qualified_records:
                f.write(json.dumps(r) + "\n")

        # Remove old bare-domain file
        src.unlink()
        print(f"[{bare_domain}] migrated {len(needs_migration)} entries → {new_path.name}")

        # Verify
        check_records = []
        with new_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        check_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        assert len(check_records) == len(records), (
            f"Record count mismatch after migration: {len(check_records)} != {len(records)}"
        )
        bad = [r for r in check_records if r.get("domain") != qualified]
        assert not bad, f"Unqualified records remain: {bad}"
        print(f"[{bare_domain}] verified: {len(check_records)} records, all domain={qualified!r}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    migrate(dry_run=dry_run)
