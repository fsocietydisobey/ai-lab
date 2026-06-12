# mnemosyne — local codebase-bound model training

The goal: a local LLM whose **weights are built on a target codebase**, driven by
Claude Code as the harness. Each project gets its own model (khimaira-expert,
jeevy-expert) — never one model blurring two codebases.

This needs **two training regimes**, in order. The pair-collection mnemosyne
already does is only the second half.

```mermaid
flowchart LR
    raw[Raw source\n(build_corpus.py)] -->|CPT\npretrain.py| cpt[CPT adapter\ncodebase in weights]
    cpt -->|merge_adapter.py| merged[Merged base]
    pairs[Q/A pairs\n(distiller → store)] -->|SFT\ntrain.py| final[Assistant adapter]
    merged --> final
    final -->|serve + point Claude Code| use[Local codebase model]
```

| Stage | Script | Objective | What it injects |
|---|---|---|---|
| 1. Continued pretraining (CPT) | `build_corpus.py` → `pretrain.py` | causal-LM on **raw source** | codebase facts/patterns into the core |
| 2. Merge | `merge_adapter.py` | fold CPT delta into base weights | — |
| 3. Instruction tuning (SFT) | `train.py` | answer-following on **Q/A pairs** | how to *use* the knowledge as an assistant |

CPT alone → autocompletes the codebase but won't follow instructions. SFT alone
(the old default) → follows instructions with no deep codebase internalization.
You need both, in this order.

## Hardware reality (this machine: 8 GB Blackwell laptop GPU, CUDA 13)

- **bf16 + LoRA fits the ≤3B tier with headroom** — no quantization needed. The
  1.5B-Coder CPT peaked 6.3 / 8.1 GB.
- **4-bit (bitsandbytes) only becomes necessary at the 7B tier** (7B bf16 = 14 GB,
  won't fit). bitsandbytes 4-bit on Blackwell+CUDA13 is **unverified** — confirm
  before any 7B run.
- A single repo (~1-2M tokens) yields **domain-adaptation**, not a from-scratch
  codebase-native model. Expect overfitting (not compute) to be the ceiling.

## Proof results (2026-06-11)

**SFT half** — `train.py`, general domain, Qwen2.5-0.5B, 300 pairs, 1 epoch:
real LoRA adapter in 48s, loss 13.67 → 7.78, no OOM.

**CPT half** — `pretrain.py`, khimaira corpus (538 files / ~1.3M tokens),
Qwen2.5-Coder-1.5B, LoRA-all-linear r=32, bf16, 1 epoch:
**held-out perplexity 6.09 → 4.21 (−30.8%)** on files the model never trained on.
The codebase generalized into the weights. 29 min, peaked 6.3 / 8.1 GB.

## Commands

```bash
# 1. assemble corpus from raw source (file-level train/eval split)
python scripts/build_corpus.py --repo /path/to/repo --out-dir corpora/<name>

# 2. continued-pretrain a code base model on it
python scripts/pretrain.py --corpus-dir corpora/<name> \
    --model Qwen/Qwen2.5-Coder-1.5B --out-name cpt-<name>

# 3. merge the CPT adapter into the base
python scripts/merge_adapter.py --base Qwen/Qwen2.5-Coder-1.5B \
    --adapter adapters/cpt-<name> --out models/merged-<name>

# 4. instruction-tune on the project's Q/A pairs (prefix glob = all project domains)
python scripts/train.py --model models/merged-<name> \
    --domain "<project>:*" --out-name sft-<name>
```

## Watching training live

Text loss prints every 10 steps to stdout. For live **curves**, both scripts write
TensorBoard event files to `runs/` when `tensorboard` is installed (graceful no-op
otherwise):

```bash
pip install tensorboard
tensorboard --logdir runs/    # open http://localhost:6006
```
Or stream the text loss: `tail -f <logfile> | grep --line-buffered "'loss'"`.
