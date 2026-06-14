#!/usr/bin/env bash
# jeevy oracle eval: CPT-only vs CPT+SFT, INSTRUCTION format. jeevy prompts test
# codebase recall + answer-shape; general prompts test catastrophic forgetting.
set -e
cd "$HOME/mnemosyne"
COMMON=(--rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
        -v "$HOME/mnemosyne:/workspace/mnemosyne" mnemosyne-train:26.05)
PROMPTS=(
  --prompt "In jeevy, how does the quote->project conversion handle deliverable ownership across the state transition?"
  --prompt "In jeevy, what is the architectural principle for keeping deliverable identity stable during quote->project conversion?"
  --prompt "In jeevy, how does the project file-manager hook handle source identity compared to the quote hook?"
  --prompt "Write a Python function that returns the nth Fibonacci number."
  --prompt "What is the difference between a Python list and a tuple?"
)
for tag_model in "CPT-only:models/cpt7b-jeevy" "CPT+SFT:models/sft7b-jeevy"; do
  tag="${tag_model%%:*}"; model="${tag_model#*:}"
  echo "################################ ${tag} (${model}) ################################"
  docker run "${COMMON[@]}" python scripts/infer.py --max-new-tokens 180 \
      --model "$model" "${PROMPTS[@]}" 2>&1 \
    | grep -vE "NVIDIA Release|Copyright|terms|legal|GOVERNING|^\*|reserved|Various|nvidia-smi|forward comp|SHMEM|kernel driver|^NOTE|See https|insufficient|docker run --gpus|UserWarning|torch_dtype|Token indices|register_constant|deprecated|Setting|loading checkpoint|^Loading|Writing model|^$"
done
echo "################################ JEEVY EVAL DONE ################################"
