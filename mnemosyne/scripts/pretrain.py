#!/usr/bin/env python3
"""Continued-pretraining (CPT) proof — adapt a code base model to a raw-source corpus.

This is the OTHER half of mnemosyne's training story. trainer.fine_tune()/train.py
do instruction-tuning on Q/A PAIRS (teaches the model how to *answer*). This does
causal-LM continued-pretraining on RAW SOURCE (puts the codebase into the weights).

The proof it produces: held-out perplexity BEFORE vs AFTER one epoch. The eval
split is whole files the model never trained on, so a perplexity DROP means the
model generalized your codebase's patterns into its weights — not memorization.

8GB-safe choices: LoRA on all linear layers (keeps optimizer state tiny),
gradient checkpointing, bf16 on Blackwell (more stable than fp16, no bitsandbytes
4-bit dependency — that path is still unverified on this GPU).

Usage:
  python scripts/pretrain.py --corpus-dir corpora/khimaira \
      --model Qwen/Qwen2.5-Coder-1.5B --block-size 1024 --epochs 1
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ADAPTERS_DIR = Path(__file__).resolve().parent.parent / "adapters"


def main() -> None:
    ap = argparse.ArgumentParser(description="CPT a code model on a raw-source corpus.")
    ap.add_argument("--corpus-dir", required=True, help="Dir with corpus_train.txt + corpus_eval.txt")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--out-name", default="cpt-khimaira")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[cpt] CUDA: {torch.cuda.is_available()} | device: {dev} | dtype: {dtype}")

    cdir = Path(args.corpus_dir)
    train_text = (cdir / "corpus_train.txt").read_text(encoding="utf-8")
    eval_text = (cdir / "corpus_eval.txt").read_text(encoding="utf-8")

    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=MODELS_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def pack(text: str) -> Dataset:
        ids = tok(text, return_attention_mask=False)["input_ids"]
        n = (len(ids) // args.block_size) * args.block_size
        blocks = [ids[i:i + args.block_size] for i in range(0, n, args.block_size)]
        return Dataset.from_dict({"input_ids": blocks})

    train_ds = pack(train_text)
    eval_ds = pack(eval_text)
    print(f"[cpt] train blocks: {len(train_ds)} | eval blocks: {len(eval_ds)} "
          f"| block={args.block_size} (~{len(train_ds)*args.block_size/1e6:.2f}M train tokens)")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=MODELS_DIR, torch_dtype=dtype, device_map="auto",
    )
    print(f"[cpt] model loaded in {time.time()-t0:.0f}s")

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # required: grad ckpt + frozen base + LoRA

    out_dir = ADAPTERS_DIR / args.out_name
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            bf16=use_bf16, fp16=not use_bf16,
            logging_steps=10, save_strategy="no", report_to=[],
            learning_rate=2e-4, warmup_ratio=0.03, lr_scheduler_type="cosine",
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )

    pre = trainer.evaluate()
    pre_ppl = math.exp(pre["eval_loss"])
    print(f"[cpt] BEFORE: eval_loss={pre['eval_loss']:.4f}  held-out ppl={pre_ppl:.2f}")

    t0 = time.time()
    trainer.train()
    print(f"[cpt] trained in {time.time()-t0:.0f}s")

    post = trainer.evaluate()
    post_ppl = math.exp(post["eval_loss"])
    delta = (pre_ppl - post_ppl) / pre_ppl * 100
    print(f"[cpt] AFTER:  eval_loss={post['eval_loss']:.4f}  held-out ppl={post_ppl:.2f}")
    print(f"[cpt] ===> held-out perplexity {pre_ppl:.2f} → {post_ppl:.2f}  ({delta:+.1f}%)")
    print("[cpt] a DROP = the codebase generalized into the weights (proof of CPT).")

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[cpt] adapter saved → {out_dir}")


if __name__ == "__main__":
    main()
