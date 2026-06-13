#!/usr/bin/env python3
"""Export SFT Q/A pairs from the local mnemosyne store to a JSONL.

Clean-corpus discipline: training runs off-machine (spark) with no store
dependency, so the refresh cycle exports the pairs locally and ships the JSONL.

  python scripts/export_sft_pairs.py --prefixes khimaira general --out corpora/sft_khimaira.jsonl

Filters out superseded pairs (kept knowledge only) and dedups.
"""

from __future__ import annotations

import argparse
import json

from mnemosyne import store


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SFT pairs from the mnemosyne store.")
    ap.add_argument("--prefixes", nargs="+", default=["khimaira", "general"],
                    help="Domain prefixes to include (e.g. khimaira general).")
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument(
        "--extra-jsonl",
        action="append",
        default=[],
        help="Curated JSONL of canonical {instruction,response} pairs to merge "
        "in (deduped). Repeatable. Use for hand-written ground-truth facts the "
        "distilled store gets wrong (e.g. ports, storage model).",
    )
    args = ap.parse_args()

    doms = [
        d for d in store.domains()
        if any(d == p or d.startswith(p + ":") for p in args.prefixes)
    ]
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    by_prefix: dict[str, int] = {}
    for d in doms:
        for p in store.load(d):
            if p.get("superseded_by"):
                continue
            ins, resp = p.get("instruction"), p.get("response")
            if not (ins and resp):
                continue
            key = (ins, resp)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"instruction": ins, "response": resp})
            pref = d.split(":", 1)[0]
            by_prefix[pref] = by_prefix.get(pref, 0) + 1

    # Merge curated ground-truth pairs last (deduped). These pin canonical facts
    # the distilled store fabricates; adding them is pure-addition (no good-pair loss).
    gt = 0
    for path in args.extra_jsonl:
        with open(path, encoding="utf-8") as ef:
            for line in ef:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                ins, resp = p.get("instruction"), p.get("response")
                if not (ins and resp):
                    continue
                key = (ins, resp)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"instruction": ins, "response": resp})
                gt += 1
    if gt:
        by_prefix["ground_truth"] = gt

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = ", ".join(f"{k}={v}" for k, v in sorted(by_prefix.items()))
    print(f"[export] {len(rows)} pairs ({counts}) → {args.out}")


if __name__ == "__main__":
    main()
