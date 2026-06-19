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

# 0. Pre-bake: distill ACTIVE, SETTLED master/lead sessions into the store so this
# cycle's in-flight knowledge (long-lived sessions that never hit Stop) is captured
# BEFORE the SFT export reads the store. Guards live in the tool: settled-only
# (idle >= 30m, so no mid-flight wrong reasoning) + masters/leads-only. Runs in the
# KHIMAIRA venv (needs the session registry + transcript + distill machinery).
# Non-fatal: the tool always exits 0 and fail-opens, so a distill hiccup (e.g. the
# distill service down) never blocks the bake.
KHIMAIRA_PY="${KHIMAIRA_REPO}/.venv/bin/python"
if [ -x "$KHIMAIRA_PY" ]; then
  log "pre-bake: distilling active settled khimaira sessions ..."
  "$KHIMAIRA_PY" -m khimaira.tools.distill_active_sessions \
    --project-root "$KHIMAIRA_REPO" --project khimaira --settle-min 30 \
    || log "active-session distill returned nonzero (non-fatal — continuing)"
else
  log "pre-bake: khimaira venv not at $KHIMAIRA_PY — skipping active-session distill"
fi

# 1. Build fresh corpus + SFT pairs locally (clean-corpus: source never ships).
log "building CPT corpus from $KHIMAIRA_REPO ..."
"$PY" scripts/build_corpus.py --repo "$KHIMAIRA_REPO" --out-dir corpora/khimaira --eval-frac 0.1 \
  || die "corpus build"
log "building clean general slice (dolly) for anti-forgetting ..."
"$PY" scripts/fetch_general.py --n 2500 --out corpora/general_clean.jsonl \
  || die "general slice"
log "exporting SFT pairs (khimaira store + clean general + ground-truth) ..."
# NOTE: --prefixes khimaira ONLY. The store's `general` domain is mislabeled
# khimaira trivia that contaminates general answers; clean general comes from
# general_clean.jsonl instead. See B-fix (2026-06).
# NOTE: escaped-bugs:khimaira is a SEPARATE prefix — the bare `khimaira` prefix
# only matches `khimaira:*`, so without naming it the captured escaped-bug pairs
# (green-tests-broke-live → seam-class → catching-test) silently never reach the
# oracle. These are curated + audit-grade (post-mortem, settled), so unlike active-
# session distills there's no mid-flight-wrong-reasoning risk in baking them.
"$PY" scripts/export_sft_pairs.py --prefixes khimaira escaped-bugs:khimaira \
  --extra-jsonl corpora/general_clean.jsonl \
  --extra-jsonl corpora/ground_truth_khimaira.jsonl \
  --out corpora/sft_khimaira.jsonl \
  || die "pairs export"

# 2. Ship corpus + scripts to spark (only text artifacts).
log "rsync corpora + scripts -> $SPARK ..."
rsync -az -e "$SSH" corpora/ "$SPARK:~/mnemosyne/corpora/"  || die "rsync corpora"
rsync -az -e "$SSH" scripts/ "$SPARK:~/mnemosyne/scripts/"  || die "rsync scripts"

# 3-6. Launch train + validate + safe-swap + reload DETACHED ON SPARK.
# The whole ~2h sequence runs on the spark via setsid+nohup, so it survives the
# laptop suspending / the SSH dropping (the original blocking heredoc died on
# laptop sleep — the exact "I'll be asleep at the scheduled time" failure). The
# laptop's only job is: build corpus -> ship -> KICK OFF -> exit. The bake then
# runs to completion on the always-on spark, writing ~/refresh.status
# (RUNNING|SUCCESS|FAILED:<stage>) + ~/refresh.log so we can check later.
log "launching detached re-bake on spark (survives laptop suspend) ..."
$SSH "$SPARK" 'setsid nohup bash ~/mnemosyne/scripts/refresh_remote.sh >> ~/refresh.log 2>&1 < /dev/null & echo "launched pid $!"' \
  || die "launch detached re-bake"

log "================= re-bake LAUNCHED (detached on spark) ================="
log "It runs ~2h independent of this laptop. Check progress any time:"
log "  ssh $SPARK 'cat ~/refresh.status; tail -5 ~/refresh.log'"
log "On SUCCESS the spark swaps the model + restarts mnemo-vllm automatically."
exit 0
