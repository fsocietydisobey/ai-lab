#!/usr/bin/env bash
# B-fix re-bake: re-run SFT ONLY (CPT unchanged) for both oracles with the
# de-contaminated pairs (clean dolly general slice replacing the mislabeled
# store-`general` khimaira trivia). Safe-swap + always keep >=1 oracle serving.
#
# GPU: one vLLM (~30GB @0.3) + one SFT (~65GB) fits the GB10; two vLLMs + SFT
# does NOT. So we stop the oracle being rebuilt, keep the OTHER serving, swap,
# restart. Net: each oracle is down only during its own ~1.5h SFT + swap.
set -uo pipefail
cd "$HOME/mnemosyne" || exit 1
DK=(docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
    -v "$HOME/mnemosyne:/workspace/mnemosyne" mnemosyne-train:26.05)

resft() {  # $1=name (jeevy|khimaira)  $2=cpt-model-dir  $3=pairs  $4=vllm-container
  local name="$1" cpt="$2" pairs="$3" cont="$4"
  echo "[resft $(date '+%T')] === $name: stop its vLLM, free GPU ==="
  docker stop "$cont" >/dev/null 2>&1 || true
  echo "[resft $(date '+%T')] SFT $name (full-FT) from $cpt -> sft7b-$name-new ..."
  "${DK[@]}" python scripts/train.py --full-ft --model "$cpt" \
    --pairs-file "$pairs" --epochs 2 --batch-size 2 --grad-accum 4 \
    --out-name "sft7b-$name-new" || { echo "[resft] $name SFT FAILED"; docker start "$cont" >/dev/null 2>&1; return 1; }
  local sz; sz=$(stat -c %s "models/sft7b-$name-new/model.safetensors" 2>/dev/null || echo 0)
  if [ "$sz" -lt 10000000000 ]; then
    echo "[resft] $name VALIDATE FAILED (new model $sz < 10GB) — keeping old"; docker start "$cont" >/dev/null 2>&1; return 1
  fi
  echo "[resft $(date '+%T')] swap (keep .prev) + restart $name vLLM ..."
  # Model dirs are root-owned (written by the training container) — the host
  # user can't mv them. Swap as root via a throwaway container; `bash -euc`
  # aborts on a failed mv so we never falsely claim success on a stale model.
  docker run --rm -v "$HOME/mnemosyne:/workspace/mnemosyne" mnemosyne-train:26.05 \
    bash -euc "
      cd /workspace/mnemosyne/models
      rm -rf sft7b-$name.prev
      if [ -d sft7b-$name ]; then mv sft7b-$name sft7b-$name.prev; fi
      mv sft7b-$name-new sft7b-$name
    " || { echo "[resft] $name SWAP FAILED — left old model live"; docker start "$cont" >/dev/null 2>&1; return 1; }
  docker start "$cont" >/dev/null 2>&1 && echo "[resft] $name vLLM restarted on de-contaminated model"
}

# jeevy first (khimaira keeps serving), then khimaira (jeevy keeps serving).
resft jeevy    models/cpt7b-jeevy    corpora/sft_jeevy.jsonl    mnemo-vllm-jeevy || { echo "[resft] ABORT after jeevy"; exit 1; }
resft khimaira models/cpt7b-khimaira corpora/sft_khimaira.jsonl mnemo-vllm        || { echo "[resft] ABORT after khimaira"; exit 1; }
echo "[resft $(date '+%T')] DONE — both oracles re-baked on clean general data"
