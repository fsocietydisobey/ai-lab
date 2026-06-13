#!/usr/bin/env python3
"""Quick local generation against a trained model — proves the assistant works.

Loads a merged/base model + optional LoRA adapter, runs one or more prompts
through it, prints the completions. This is the cheap end-to-end check before
standing up a serving stack: does the composed (CPT→SFT) model actually produce
coherent, codebase-aware answers?

Usage:
  # composed model = merged CPT base + SFT adapter
  python scripts/infer.py --model models/merged-khimaira-cpt \
      --adapter adapters/sft-khimaira \
      --prompt "What does the roster_recovery module do in khimaira?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

DEFAULT_PROMPTS = [
    "What does the roster_recovery module do in khimaira?",
    "How does khimaira prevent duplicate kitty window titles from breaking title-matching?",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run generation against a trained model.")
    ap.add_argument("--model", required=True, help="Base or merged model dir/name")
    ap.add_argument("--adapter", default="", help="Optional LoRA adapter to stack on top")
    ap.add_argument("--prompt", action="append", help="Prompt (repeatable; defaults to a builtin set)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--raw", action="store_true",
                    help="Raw completion (no ### Instruction wrapper) — for CPT/base "
                         "models that have not been SFT'd into instruction-following yet.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model, cache_dir=MODELS_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=MODELS_DIR, torch_dtype=dtype, device_map="auto",
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompts = args.prompt or DEFAULT_PROMPTS
    for p in prompts:
        # raw completion for CPT/base models; instruction wrapper for SFT'd models
        text = p if args.raw else f"### Instruction:\n{p}\n\n### Response:\n"
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        completion = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n{'='*70}\nPROMPT: {p}\n{'-'*70}\n{completion.strip()}\n")


if __name__ == "__main__":
    main()
