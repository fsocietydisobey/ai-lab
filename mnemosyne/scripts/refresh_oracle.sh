#!/usr/bin/env bash
# Nightly khimaira-oracle re-bake. Runs LOCALLY (needs the source + mnemosyne
# store); trains on spark. Safe-swap: the live model is replaced ONLY if the new
# bake trains and validates, so a failed/OOM bake never takes the oracle down.
#
#   build fresh corpus + SFT pairs  →  rsync to spark  →  CPT → SFT(new)
#   →  validate  →  swap live model (keep .prev)  →  reload vLLM
#
# Wire via a systemd timer (OnCalendar 03:00, Persistent=true) so a suspended
# laptop catches up on wake. ~2h wall-clock; zero API cost.
set -uo pipefail

LOCAL_MNEMO="${MNEMO_DIR:-$HOME/dev/ai-lab/mnemosyne}"
KHIMAIRA_REPO="${KHIMAIRA_REPO:-$HOME/dev/khimaira}"
SPARK="${SPARK_HOST:-spark}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes"
PY="$LOCAL_MNEMO/.venv/bin/python"

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "FATAL: $*"; exit 1; }

log "================= oracle re-bake START ================="
cd "$LOCAL_MNEMO" || die "no mnemosyne dir at $LOCAL_MNEMO"

# 1. Build fresh corpus + SFT pairs locally (clean-corpus: source never ships).
log "building CPT corpus from $KHIMAIRA_REPO ..."
"$PY" scripts/build_corpus.py --repo "$KHIMAIRA_REPO" --out-dir corpora/khimaira --eval-frac 0.1 \
  || die "corpus build"
log "exporting SFT pairs from mnemosyne store ..."
"$PY" scripts/export_sft_pairs.py --prefixes khimaira general --out corpora/sft_khimaira.jsonl \
  || die "pairs export"

# 2. Ship corpus + scripts to spark (only text artifacts).
log "rsync corpora + scripts -> $SPARK ..."
rsync -az -e "$SSH" corpora/ "$SPARK:~/mnemosyne/corpora/"  || die "rsync corpora"
rsync -az -e "$SSH" scripts/ "$SPARK:~/mnemosyne/scripts/"  || die "rsync scripts"

# 3-6. Train + validate + safe-swap + reload, all on spark.
log "training on spark (CPT ~32m + SFT ~1.5h) ..."
$SSH "$SPARK" 'bash -s' <<'REMOTE'
set -uo pipefail
cd ~/mnemosyne || exit 1
DK=(docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
    -v "$HOME/mnemosyne:/workspace/mnemosyne" mnemosyne-train:26.05)
echo "[spark] CPT (full-FT) ..."
"${DK[@]}" python scripts/pretrain.py --corpus-dir corpora/khimaira \
  --model Qwen/Qwen2.5-Coder-7B --full-ft --epochs 1 --block-size 1024 \
  --batch-size 1 --grad-accum 8 --out-name cpt7b-khimaira || { echo "[spark] CPT FAILED"; exit 1; }
echo "[spark] SFT (full-FT) -> sft7b-khimaira-new ..."
"${DK[@]}" python scripts/train.py --full-ft --model models/cpt7b-khimaira \
  --pairs-file corpora/sft_khimaira.jsonl --epochs 2 --batch-size 2 --grad-accum 4 \
  --out-name sft7b-khimaira-new || { echo "[spark] SFT FAILED"; exit 1; }
SZ=$(stat -c %s ~/mnemosyne/models/sft7b-khimaira-new/model.safetensors 2>/dev/null || echo 0)
if [ "$SZ" -lt 10000000000 ]; then echo "[spark] VALIDATE FAILED: new model too small ($SZ)"; exit 1; fi
echo "[spark] swap (keep .prev) + reload vLLM ..."
rm -rf ~/mnemosyne/models/sft7b-khimaira.prev
[ -d ~/mnemosyne/models/sft7b-khimaira ] && mv ~/mnemosyne/models/sft7b-khimaira ~/mnemosyne/models/sft7b-khimaira.prev
mv ~/mnemosyne/models/sft7b-khimaira-new ~/mnemosyne/models/sft7b-khimaira
docker restart mnemo-vllm >/dev/null && echo "[spark] vLLM restarted on new model"
REMOTE
RC=$?

if [ "$RC" -eq 0 ]; then
  log "================= oracle re-bake SUCCESS ================="
else
  log "===== oracle re-bake FAILED (rc=$RC) — LIVE MODEL UNCHANGED ====="
fi
exit "$RC"
