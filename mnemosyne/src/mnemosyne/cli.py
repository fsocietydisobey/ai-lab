"""Mnemosyne CLI."""

from __future__ import annotations

import sys
import uvicorn


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "serve":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8766
        uvicorn.run("mnemosyne.api.main:app", host="127.0.0.1", port=port, reload=False)
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: mnemosyne serve [port]")
        sys.exit(1)
