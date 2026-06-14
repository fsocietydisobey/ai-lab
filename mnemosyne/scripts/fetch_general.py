#!/usr/bin/env python3
"""Build a CLEAN general-instruction SFT slice for anti-forgetting.

The mnemosyne store's `general` domain is mislabeled codebase trivia (it bleeds
khimaira specifics into general answers — confirmed on the serving path). Real
anti-forgetting needs genuinely general instruction data. This pulls a
deterministic sample from databricks-dolly-15k (Apache-2.0, human-written:
brainstorming, QA, classification, generation, summarization) and writes it as
{instruction, response} JSONL — no codebase content, so it teaches "stay
general" without dragging either codebase into general answers.

  python scripts/fetch_general.py --n 2500 --out corpora/general_clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a clean general SFT slice.")
    ap.add_argument("--n", type=int, default=2500, help="How many pairs to sample.")
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--seed", type=int, default=13, help="Deterministic sample seed.")
    args = ap.parse_args()

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    rows: list[dict] = []
    for r in ds:
        instr = (r.get("instruction") or "").strip()
        ctx = (r.get("context") or "").strip()
        resp = (r.get("response") or "").strip()
        if not (instr and resp):
            continue
        # Fold context into the instruction so the pair is self-contained.
        if ctx:
            instr = f"{instr}\n\nContext:\n{ctx}"
        rows.append({"instruction": instr, "response": resp})

    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[general] {len(rows)} clean general pairs (seed={args.seed}) → {args.out}")


if __name__ == "__main__":
    main()
