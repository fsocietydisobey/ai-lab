#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model, producing standalone weights.

This is the hinge of the full recipe. CPT (pretrain.py) produces a LoRA adapter
that holds the codebase knowledge as a low-rank delta on top of the base. To then
instruction-tune ON TOP of that knowledge, the CPT delta must become part of the
base weights — otherwise the SFT LoRA would train against the un-adapted base and
the codebase knowledge wouldn't be in the starting point.

merge_and_unload() folds the adapter into the weights and returns a plain model
we save to disk. train.py then points --model at that merged dir.

Usage:
  python scripts/merge_adapter.py --base Qwen/Qwen2.5-Coder-1.5B \
      --adapter adapters/cpt-khimaira --out models/merged-khimaira-cpt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge a LoRA adapter into base weights.")
    ap.add_argument("--base", required=True, help="Base HF model the adapter was trained on")
    ap.add_argument("--adapter", required=True, help="Path to the LoRA adapter dir")
    ap.add_argument("--out", required=True, help="Output dir for the merged model")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.base, cache_dir=MODELS_DIR, torch_dtype=dtype, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    print(f"[merge] loaded base + adapter in {time.time()-t0:.0f}s")

    merged = model.merge_and_unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(out)
    print(f"[merge] ✅ merged model saved → {out}")


if __name__ == "__main__":
    main()
