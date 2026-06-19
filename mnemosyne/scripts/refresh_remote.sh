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
# ALL oracle vLLM containers — stopped for the bake to free the unified pool, then
# restarted at the end. A full-FT 7.6B train + its save-step memory spike does NOT
# fit alongside the always-on oracles (they reserve ~75/121GB), so the save was
# SIGKILLed at "Writing model shards" every run. Stopping them is the weekly
# maintenance window; mnemosyne_ask fail-opens while they're down.
ALL_VLLM="${ALL_VLLM_CONTAINERS:-mnemo-vllm mnemo-vllm-jeevy}"
STATUS="$HOME/refresh-$ORACLE.status"

cd ~/mnemosyne || { echo "no ~/mnemosyne" >&2; echo "FAILED:nodir" > "$STATUS"; exit 1; }

# Restart the oracles unconditionally on exit (success OR failure) so a failed bake
# never strands them down. Safe-swap means a failed model was never swapped in, so
# each container restarts on its still-valid model dir.
_restart_oracles() {
  echo "[remote:$ORACLE] restarting vLLM oracles: $ALL_VLLM"
  docker start $ALL_VLLM >/dev/null 2>&1 || true
}
echo "RUNNING $(date '+%F %T')" > "$STATUS"
fail() {
  echo "FAILED:$1 $(date '+%F %T')" > "$STATUS"
  echo "[remote:$ORACLE] FAILED at $1"
  _restart_oracles
  exit 1
}

echo "[remote:$ORACLE] freeing GPU pool — stopping vLLM oracles: $ALL_VLLM"
docker stop $ALL_VLLM >/dev/null 2>&1 || true

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
# Restart ALL oracles. The baked one ($VLLM_CONTAINER) reloads from its now-swapped
# model dir → serves the new model; the sibling restarts on its unchanged model.
_restart_oracles
echo "[remote:$ORACLE] $VLLM_CONTAINER now serving the newly-baked model"

echo "SUCCESS $(date '+%F %T')" > "$STATUS"
echo "[remote:$ORACLE $(date '+%T')] ================= re-bake SUCCESS ================="
