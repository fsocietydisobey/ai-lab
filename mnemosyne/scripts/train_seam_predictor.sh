#!/usr/bin/env bash
# Phase 3 of the escaped-bugs flywheel: fine-tune a SEAM-PREDICTOR on the
# escaped-bugs corpus. Input pair: (situation: symptom + what the test mocked)
# -> (seam-class + catching-test pattern). The /khimaira-distill-bugs command
# stores these directly in the mnemosyne store (escaped-bugs:<project>), so this
# just exports + trains — same infra as the codebase oracle.
#
# SIZE-GATED: refuses to train below MIN_PAIRS. A handful of examples can't train
# a generalizing model — retrieval (/khimaira-recall-bugs) is the right tool until
# the corpus is large. This gate is the "train once the dataset is big enough"
# discipline; running it early is a no-op that tells you how far off you are.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MIN_PAIRS="${MIN_PAIRS:-50}"          # floor to even attempt a fine-tune
PREFIX="${SEAM_PREFIX:-escaped-bugs}"  # store domain prefix to pull
OUT="corpora/sft_seam.jsonl"
PY="${PY:-.venv/bin/python}"; [ -x "$PY" ] || PY=python3

echo "[seam] exporting escaped-bugs pairs (prefix=$PREFIX) ..."
"$PY" scripts/export_sft_pairs.py --prefixes "$PREFIX" --out "$OUT" || { echo "[seam] export FAILED"; exit 1; }

N=$(wc -l < "$OUT" 2>/dev/null | tr -d ' ')
N="${N:-0}"
echo "[seam] corpus size: $N pairs (floor to train: $MIN_PAIRS)"

if [ "$N" -lt "$MIN_PAIRS" ]; then
  cat <<EOF
[seam] ⛔ NOT TRAINING — corpus too small ($N/$MIN_PAIRS).
       This is the size-gate working as designed, not a failure.
       Keep capturing escapes with /khimaira-distill-bugs; meanwhile
       /khimaira-recall-bugs (retrieval) already gives value at this size.
       Re-run when the corpus reaches $MIN_PAIRS audit-grade pairs.
EOF
  exit 0
fi

echo "[seam] ✅ corpus big enough — training seam-predictor (LoRA on Qwen2.5-Coder-7B) ..."
# LoRA (not full-FT): the seam-predictor is a narrow specialization; a light
# adapter over the base generalizes better than a full-FT on a small set and is
# far cheaper. Override --model to a codebase CPT checkpoint (e.g.
# models/cpt7b-khimaira) to start from codebase-aware weights.
DK=(docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
    -v "$HOME/mnemosyne:/workspace/mnemosyne" mnemosyne-train:26.05)
"${DK[@]}" python scripts/train.py \
  --model "${SEAM_BASE:-Qwen/Qwen2.5-Coder-7B}" \
  --pairs-file "$OUT" --epochs 3 --batch-size 2 --grad-accum 4 \
  --out-name "${SEAM_OUT:-seam-predictor}" \
  || { echo "[seam] training FAILED"; exit 1; }
echo "[seam] DONE — seam-predictor at models/${SEAM_OUT:-seam-predictor}"
echo "[seam] serve it like the oracles (serve_oracles.sh pattern) and route a"
echo "[seam] /khimaira-recall-bugs --model seam-predictor path once live."
