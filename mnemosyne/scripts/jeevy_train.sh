#!/usr/bin/env bash
# Train the jeevy oracle: CPT (full-FT) -> SFT (full-FT). Runs on spark; coexists
# with the live khimaira vLLM (which must be at LOW --gpu-memory-utilization so
# both fit the 128GB unified pool). ~5h on the GB10. Zero API cost.
set -uo pipefail
cd ~/mnemosyne || exit 1
DK=(docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
    -v "$HOME/mnemosyne:/workspace/mnemosyne" mnemosyne-train:26.05)

echo "[jeevy $(date '+%T')] CPT (full-FT) on the 34.8MB jeevy corpus ..."
"${DK[@]}" python scripts/pretrain.py --corpus-dir corpora/jeevy \
  --model Qwen/Qwen2.5-Coder-7B --full-ft --epochs 1 --block-size 1024 \
  --batch-size 1 --grad-accum 8 --out-name cpt7b-jeevy \
  || { echo "[jeevy] CPT FAILED"; exit 1; }

echo "[jeevy $(date '+%T')] SFT (full-FT) on jeevy+general pairs ..."
"${DK[@]}" python scripts/train.py --full-ft --model models/cpt7b-jeevy \
  --pairs-file corpora/sft_jeevy.jsonl --epochs 2 --batch-size 2 --grad-accum 4 \
  --out-name sft7b-jeevy \
  || { echo "[jeevy] SFT FAILED"; exit 1; }

echo "[jeevy $(date '+%T')] DONE — model at models/sft7b-jeevy"
