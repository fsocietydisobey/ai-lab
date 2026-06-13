#!/usr/bin/env bash
# Base 7B vs CPT'd 7B, raw completion. Codebase prompts = absorption test;
# general-coding prompts = catastrophic-forgetting check.
set -e
cd "$HOME/mnemosyne"
COMMON=(--rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
        -v "$HOME/mnemosyne:/workspace/mnemosyne" mnemosyne-train:26.05)
PROMPTS=(
  --prompt "# khimaira monitor: the roster_recovery module is responsible for"
  --prompt "In khimaira, the auto_dispatch_loop froze in production. The root cause was that uvloop"
  --prompt "# In khimaira, chat_reseat_master re-seats a new session as the roster master after the previous master"
  --prompt "def binary_search(sorted_list, target):"
  --prompt "# Python function to check whether a string is a palindrome"
)
for tag_model in "BASE:Qwen/Qwen2.5-Coder-7B" "CPT:models/cpt7b-khimaira"; do
  tag="${tag_model%%:*}"; model="${tag_model#*:}"
  echo "################################ ${tag} (${model}) ################################"
  docker run "${COMMON[@]}" python scripts/infer.py --raw --max-new-tokens 130 \
      --model "$model" "${PROMPTS[@]}" 2>&1 \
    | grep -vE "NVIDIA Release|Copyright|terms|legal|GOVERNING|^\*|reserved|Various|nvidia-smi|forward comp|SHMEM|kernel driver|^NOTE|See https|insufficient|docker run --gpus|UserWarning|torch_dtype|Token indices|register_constant|deprecated|Setting|loading checkpoint|^Loading|^$"
done
echo "################################ EVAL DONE ################################"
