"""Mnemosyne CLI."""

from __future__ import annotations

import argparse
import sys
import uvicorn


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "serve":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8766
        uvicorn.run("mnemosyne.api.main:app", host="127.0.0.1", port=port, reload=False)
    elif cmd == "supersede":
        _cmd_supersede(sys.argv[2:])
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: mnemosyne serve [port]")
        print("       mnemosyne supersede --domain DOMAIN --id ID --by BY [--reason REASON]")
        sys.exit(1)


def _cmd_supersede(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="mnemosyne supersede",
        description="Mark a training pair as superseded by another.",
    )
    parser.add_argument("--domain", required=True, help="Domain key (e.g. khimaira:backend)")
    parser.add_argument("--id", dest="target_id", required=True, help="ID of the pair to supersede")
    parser.add_argument("--by", dest="by_id", required=True, help="ID of the superseding pair")
    parser.add_argument("--reason", default="", help="Optional human-readable reason")
    parsed = parser.parse_args(args)

    from mnemosyne import store

    try:
        store.supersede(parsed.domain, parsed.target_id, parsed.by_id, parsed.reason)
        print(f"Superseded {parsed.target_id!r} by {parsed.by_id!r} in domain {parsed.domain!r}.")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
