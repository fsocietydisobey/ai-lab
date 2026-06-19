#!/usr/bin/env bash
# Spark-side oracle re-bake: CPT -> SFT -> validate -> safe-swap -> reload vLLM.
# Parameterized by ORACLE (default "khimaira") so the khimaira AND jeevy oracles
# share ONE remote bake. Launched DETACHED by refresh_oracle.sh / refresh_jeevy.sh
# (setsid/nohup) so the multi-hour bake survives the laptop suspending / the SSH
# dropping — the whole sequence runs on the always-on spark.
#
# Per-oracle status + log so the two runs never clobber each other:
#   ~/refresh-$ORACLE.status   RUNNING | SUCCESS | FAILED:<stage>
#   ~/refresh-$ORACLE.log      (the launcher redirects stdout/err here)
set -uo pipefail

ORACLE="${ORACLE:-khimaira}"
CPT_CORPUS="${CPT_CORPUS:-corpora/$ORACLE}"
CPT_OUT="${CPT_OUT:-cpt7b-$ORACLE}"
SFT_PAIRS="${SFT_PAIRS:-corpora/sft_$ORACLE.jsonl}"
SFT_OUT="${SFT_OUT:-sft7b-$ORACLE}"
VLLM_CONTAINER="${VLLM_CONTAINER:-mnemo-vllm}"
STATUS="$HOME/refresh-$ORACLE.status"

cd ~/mnemosyne || { echo "no ~/mnemosyne" >&2; echo "FAILED:nodir" > "$STATUS"; exit 1; }

echo "RUNNING $(date '+%F %T')" > "$STATUS"
fail() { echo "FAILED:$1 $(date '+%F %T')" > "$STATUS"; echo "[remote:$ORACLE] FAILED at $1"; exit 1; }

DK=(docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
    -v "$HOME/mnemosyne:/workspace/mnemosyne" mnemosyne-train:26.05)

echo "[remote:$ORACLE $(date '+%T')] CPT (full-FT) on $CPT_CORPUS -> $CPT_OUT ..."
"${DK[@]}" python scripts/pretrain.py --corpus-dir "$CPT_CORPUS" \
  --model Qwen/Qwen2.5-Coder-7B --full-ft --epochs 1 --block-size 1024 \
  --batch-size 1 --grad-accum 8 --out-name "$CPT_OUT" || fail CPT

echo "[remote:$ORACLE $(date '+%T')] SFT (full-FT) $SFT_PAIRS -> ${SFT_OUT}-new ..."
"${DK[@]}" python scripts/train.py --full-ft --model "models/$CPT_OUT" \
  --pairs-file "$SFT_PAIRS" --epochs 2 --batch-size 2 --grad-accum 4 \
  --out-name "${SFT_OUT}-new" || fail SFT

SZ=$(stat -c %s "$HOME/mnemosyne/models/${SFT_OUT}-new/model.safetensors" 2>/dev/null || echo 0)
[ "$SZ" -ge 10000000000 ] || fail "validate(new model $SZ < 10GB)"

echo "[remote:$ORACLE $(date '+%T')] swap (keep .prev) + reload $VLLM_CONTAINER ..."
# Model dirs are ROOT-owned (training container) — swap via a root container. The
# inner script is DOUBLE-quoted so $SFT_OUT interpolates from the outer shell; the
# value is a literal model name (no user input), so this is safe.
docker run --rm -v "$HOME/mnemosyne:/workspace/mnemosyne" mnemosyne-train:26.05 \
  bash -euc "
    cd /workspace/mnemosyne/models
    rm -rf ${SFT_OUT}.prev
    if [ -d ${SFT_OUT} ]; then mv ${SFT_OUT} ${SFT_OUT}.prev; fi
    mv ${SFT_OUT}-new ${SFT_OUT}
  " || fail swap
docker restart "$VLLM_CONTAINER" >/dev/null \
  && echo "[remote:$ORACLE] $VLLM_CONTAINER restarted on new model"

echo "SUCCESS $(date '+%F %T')" > "$STATUS"
echo "[remote:$ORACLE $(date '+%T')] ================= re-bake SUCCESS ================="
