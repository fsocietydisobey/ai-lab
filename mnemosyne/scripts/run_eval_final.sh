#!/usr/bin/env bash
# CPT-only vs CPT+SFT, INSTRUCTION format (the SFT'd model should now *answer*).
# khimaira prompts = does it answer the codebase correctly; general = forgetting.
set -e
cd "$HOME/mnemosyne"
COMMON=(--rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
        -v "$HOME/mnemosyne:/workspace/mnemosyne" mnemosyne-train:26.05)
PROMPTS=(
  --prompt "What does the roster_recovery module do in khimaira?"
  --prompt "In khimaira, what is chat_reseat_master and when do you use it?"
  --prompt "Why did khimaira's auto_dispatch loop freeze in production, and how was it fixed?"
  --prompt "Write a Python function that returns the nth Fibonacci number."
  --prompt "What is the difference between a Python list and a tuple?"
)
for tag_model in "CPT-only:models/cpt7b-khimaira" "CPT+SFT:models/sft7b-khimaira"; do
  tag="${tag_model%%:*}"; model="${tag_model#*:}"
  echo "################################ ${tag} (${model}) ################################"
  docker run "${COMMON[@]}" python scripts/infer.py --max-new-tokens 180 \
      --model "$model" "${PROMPTS[@]}" 2>&1 \
    | grep -vE "NVIDIA Release|Copyright|terms|legal|GOVERNING|^\*|reserved|Various|nvidia-smi|forward comp|SHMEM|kernel driver|^NOTE|See https|insufficient|docker run --gpus|UserWarning|torch_dtype|Token indices|register_constant|deprecated|Setting|loading checkpoint|^Loading|Writing model|^$"
done
echo "################################ FINAL EVAL DONE ################################"
