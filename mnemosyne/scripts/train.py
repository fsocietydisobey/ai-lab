#!/usr/bin/env python3
"""Runnable LoRA training entrypoint (the missing CLI for Phase 2).

mnemosyne shipped trainer.fine_tune() as a library function but never wired a
way to RUN it. This is that entrypoint, with two pragmatic departures from the
hardcoded trainer for THIS machine (8GB Blackwell laptop GPU on CUDA 13):

  1. --no-4bit (default for small models): bitsandbytes 4-bit on a Blackwell /
     CUDA-13 GPU is bleeding-edge and may not load. Small models (0.5B-1.5B) fit
     8GB in fp16 anyway, so we skip quantization for the proof-of-concept and
     only need it for the 7B tier.
  2. --max-pairs / --epochs: bound a smoke run so we can PROVE the GPU pipeline
     end-to-end in minutes before committing to a full multi-hour fine-tune.

Usage:
  python scripts/train.py --domain general --model Qwen/Qwen2.5-0.5B-Instruct \
      --max-pairs 300 --epochs 1            # fast proof-of-concept
  python scripts/train.py --domain general --model Qwen/Qwen2.5-7B-Instruct \
      --4bit --epochs 3                     # the real thing (needs 4-bit to fit 8GB)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ADAPTERS_DIR = Path(__file__).resolve().parent.parent / "adapters"


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune a domain memory model.")
    ap.add_argument("--domain", required=True,
                    help="Domain key ('khimaira:backend'), or prefix glob ('khimaira:*') to "
                         "combine every matching project domain into one SFT set")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="Base HF model OR a local path (e.g. a merged CPT model)")
    ap.add_argument("--max-pairs", type=int, default=0, help="Cap training pairs (0 = all)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--out-name", default="", help="Adapter dir name (default: derived from --domain)")
    ap.add_argument("--4bit", dest="use_4bit", action="store_true", help="4-bit quantize (needed for 7B on 8GB)")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from mnemosyne import store

    print(f"[train] CUDA available: {torch.cuda.is_available()} | device: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    if args.domain.endswith("*"):
        prefix = args.domain[:-1]
        matched = sorted(d for d in store.domains() if d.startswith(prefix))
        pairs = [p for d in matched for p in store.load(d)]
        print(f"[train] glob '{args.domain}' matched {len(matched)} domains: {', '.join(matched)}")
    else:
        pairs = store.load(args.domain)
    if not pairs:
        raise SystemExit(f"No training data for domain '{args.domain}'")
    if args.max_pairs:
        pairs = pairs[-args.max_pairs:]
    print(f"[train] domain={args.domain} pairs={len(pairs)} model={args.model} "
          f"4bit={args.use_4bit} epochs={args.epochs}")

    texts = [
        f"### Instruction:\n{p['instruction']}\n\n### Response:\n{p['response']}"
        for p in pairs
    ]
    dataset = Dataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=MODELS_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(cache_dir=MODELS_DIR, torch_dtype=torch.float16, device_map="auto")
    if args.use_4bit:
        load_kwargs["load_in_4bit"] = True
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    print(f"[train] model loaded in {time.time()-t0:.0f}s")

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    model.print_trainable_parameters()

    def tokenize(batch: dict) -> dict:
        out = tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    out_name = args.out_name or args.domain.replace(":", "_").replace("*", "all")
    out_dir = ADAPTERS_DIR / out_name
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to=[],
        ),
        train_dataset=dataset.map(tokenize, batched=True, remove_columns=["text"]),
    )
    t0 = time.time()
    trainer.train()
    print(f"[train] training done in {time.time()-t0:.0f}s")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[train] ✅ LoRA adapter saved → {out_dir}")


if __name__ == "__main__":
    main()
