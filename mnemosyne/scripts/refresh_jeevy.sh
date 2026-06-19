#!/usr/bin/env bash
# Weekly jeevy-oracle re-bake — the jeevy twin of refresh_oracle.sh. Runs LOCALLY
# (needs the jeevy source + mnemosyne store); trains on spark. Safe-swap: the live
# jeevy model is replaced ONLY if the new bake trains + validates.
#
#   distill active jeevy sessions  →  build jeevy corpus + SFT pairs (incl.
#   escaped-bugs:jeevy)  →  rsync to spark  →  CPT → SFT(new) → validate →
#   swap live model (keep .prev) → reload mnemo-vllm-jeevy
#
# Staggered OFF the khimaira Friday bake (jeevy timer = Saturday) so the two
# multi-hour trains never contend for the GB10's unified pool. ~5h; zero API cost.
set -uo pipefail

LOCAL_MNEMO="${MNEMO_DIR:-$HOME/dev/ai-lab/mnemosyne}"
JEEVY_REPO="${JEEVY_REPO:-$HOME/work/jeevy_portal}"
KHIMAIRA_REPO="${KHIMAIRA_REPO:-$HOME/dev/khimaira}"  # for the distiller venv only
SPARK="${SPARK_HOST:-spark}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes"
PY="$LOCAL_MNEMO/.venv/bin/python"

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "FATAL: $*"; exit 1; }

log "================= jeevy oracle re-bake START ================="
cd "$LOCAL_MNEMO" || die "no mnemosyne dir at $LOCAL_MNEMO"

# 0. Pre-bake: distill ACTIVE, SETTLED jeevy master/lead sessions. The jeevy roster
# master is "muther" (not *-0/*master* shaped), so it must be declared explicitly via
# --master-names. Runs in the KHIMAIRA venv (the tool ships there). Non-fatal.
KHIMAIRA_PY="${KHIMAIRA_REPO}/.venv/bin/python"
if [ -x "$KHIMAIRA_PY" ]; then
  log "pre-bake: distilling active settled jeevy sessions ..."
  "$KHIMAIRA_PY" -m khimaira.tools.distill_active_sessions \
    --project-root "$JEEVY_REPO" --project jeevy --settle-min 30 \
    --master-names muther,janice \
    || log "active-session distill returned nonzero (non-fatal — continuing)"
else
  log "pre-bake: khimaira venv not at $KHIMAIRA_PY — skipping active-session distill"
fi

# 1. Build fresh jeevy corpus + SFT pairs locally (clean-corpus: source never ships).
log "building CPT corpus from $JEEVY_REPO ..."
"$PY" scripts/build_corpus.py --repo "$JEEVY_REPO" --out-dir corpora/jeevy --eval-frac 0.1 \
  || die "corpus build"
log "building clean general slice (dolly) for anti-forgetting ..."
"$PY" scripts/fetch_general.py --n 2500 --out corpora/general_clean.jsonl \
  || die "general slice"
# --prefixes jeevy escaped-bugs:jeevy: the bare `jeevy` prefix only matches `jeevy:*`,
# so escaped-bugs:jeevy must be named explicitly or the captured seam-class pairs
# never reach the oracle. No jeevy ground-truth file exists; clean general comes from
# general_clean.jsonl. The mislabeled bare `general` domain is excluded (B-fix).
log "exporting SFT pairs (jeevy store + escaped-bugs:jeevy + clean general) ..."
"$PY" scripts/export_sft_pairs.py --prefixes jeevy escaped-bugs:jeevy \
  --extra-jsonl corpora/general_clean.jsonl \
  --out corpora/sft_jeevy.jsonl \
  || die "pairs export"

# 2. Ship corpus + scripts to spark (only text artifacts).
log "rsync corpora + scripts -> $SPARK ..."
rsync -az -e "$SSH" corpora/ "$SPARK:~/mnemosyne/corpora/"  || die "rsync corpora"
rsync -az -e "$SSH" scripts/ "$SPARK:~/mnemosyne/scripts/"  || die "rsync scripts"

# 3-6. Launch train + validate + safe-swap + reload DETACHED ON SPARK, parameterized
# for jeevy: ORACLE=jeevy (→ corpora/jeevy, sft_jeevy.jsonl, sft7b-jeevy, and the
# ~/refresh-jeevy.status/.log files) + VLLM_CONTAINER=mnemo-vllm-jeevy (the :18001
# server). Detached so the ~5h bake survives the laptop suspending.
log "launching detached jeevy re-bake on spark (survives laptop suspend) ..."
$SSH "$SPARK" 'ORACLE=jeevy VLLM_CONTAINER=mnemo-vllm-jeevy setsid nohup bash ~/mnemosyne/scripts/refresh_remote.sh >> ~/refresh-jeevy.log 2>&1 < /dev/null & echo "launched pid $!"' \
  || die "launch detached jeevy re-bake"

log "================= jeevy re-bake LAUNCHED (detached on spark) ================="
log "It runs ~5h independent of this laptop. Check progress any time:"
log "  ssh $SPARK 'cat ~/refresh-jeevy.status; tail -5 ~/refresh-jeevy.log'"
log "On SUCCESS the spark swaps sft7b-jeevy + restarts mnemo-vllm-jeevy automatically."
exit 0
