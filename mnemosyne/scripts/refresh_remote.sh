#!/usr/bin/env bash
# Spark-side oracle re-bake: CPT -> SFT -> validate -> safe-swap -> reload vLLM.
# Launched DETACHED by refresh_oracle.sh (nohup/setsid) so the ~2h bake survives
# the laptop suspending / the SSH dropping — the whole sequence runs on the spark.
# Logs to ~/refresh.log; writes ~/refresh.status (RUNNING|SUCCESS|FAILED:<stage>)
# so the laptop can check completion on next wake without a held connection.
set -uo pipefail
cd ~/mnemosyne || { echo "no ~/mnemosyne" >&2; echo "FAILED:nodir" > ~/refresh.status; exit 1; }

echo "RUNNING $(date '+%F %T')" > ~/refresh.status
fail() { echo "FAILED:$1 $(date '+%F %T')" > ~/refresh.status; echo "[remote] FAILED at $1"; exit 1; }

DK=(docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
    -v "$HOME/mnemosyne:/workspace/mnemosyne" mnemosyne-train:26.05)

echo "[remote $(date '+%T')] CPT (full-FT) ..."
"${DK[@]}" python scripts/pretrain.py --corpus-dir corpora/khimaira \
  --model Qwen/Qwen2.5-Coder-7B --full-ft --epochs 1 --block-size 1024 \
  --batch-size 1 --grad-accum 8 --out-name cpt7b-khimaira || fail CPT

echo "[remote $(date '+%T')] SFT (full-FT) -> sft7b-khimaira-new ..."
"${DK[@]}" python scripts/train.py --full-ft --model models/cpt7b-khimaira \
  --pairs-file corpora/sft_khimaira.jsonl --epochs 2 --batch-size 2 --grad-accum 4 \
  --out-name sft7b-khimaira-new || fail SFT

SZ=$(stat -c %s ~/mnemosyne/models/sft7b-khimaira-new/model.safetensors 2>/dev/null || echo 0)
[ "$SZ" -ge 10000000000 ] || fail "validate(new model $SZ < 10GB)"

echo "[remote $(date '+%T')] swap (keep .prev) + reload vLLM ..."
# Model dirs are ROOT-owned (training container) — swap via a root container.
docker run --rm -v "$HOME/mnemosyne:/workspace/mnemosyne" mnemosyne-train:26.05 \
  bash -euc '
    cd /workspace/mnemosyne/models
    rm -rf sft7b-khimaira.prev
    if [ -d sft7b-khimaira ]; then mv sft7b-khimaira sft7b-khimaira.prev; fi
    mv sft7b-khimaira-new sft7b-khimaira
  ' || fail swap
docker restart mnemo-vllm >/dev/null && echo "[remote] vLLM restarted on new model"

echo "SUCCESS $(date '+%F %T')" > ~/refresh.status
echo "[remote $(date '+%T')] ================= re-bake SUCCESS ================="
