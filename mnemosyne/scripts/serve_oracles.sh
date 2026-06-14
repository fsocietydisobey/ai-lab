#!/usr/bin/env bash
# Canonical launch for the two mnemosyne codebase oracles on the Spark (vLLM,
# host-net, FP8). Idempotent: rm -f + run. Containers carry --restart
# unless-stopped so they survive reboots; this script is for first launch or a
# deliberate recreate. Run ON the spark (or via ssh spark 'bash -s' < this).
#
# FP8 NOTE: --quantization fp8 (Blackwell-native CutlassFP8 kernel, ~2x tok/s,
# ~half weight memory) REQUIRES --enforce-eager on the GB10 — without it,
# engine-core init crash-loops during cudagraph capture. Do not drop it.
set -uo pipefail
IMG=nvcr.io/nvidia/vllm:26.05.post1-py3
MNT="$HOME/mnemosyne:/workspace/mnemosyne"

launch() {  # $1=container $2=model-dir $3=served-name $4=port
  docker rm -f "$1" >/dev/null 2>&1 || true
  docker run -d --name "$1" --restart unless-stopped \
    --network host --ipc host --gpus all -v "$MNT" "$IMG" \
    vllm serve "/workspace/mnemosyne/models/$2" \
      --served-model-name "$3" --port "$4" --max-model-len 4096 \
      --gpu-memory-utilization 0.3 --quantization fp8 --enforce-eager \
      --chat-template /workspace/mnemosyne/sft_chat_template.jinja >/dev/null \
    && echo "launched $1 ($3 on :$4, fp8+eager)"
}

launch mnemo-vllm       sft7b-khimaira khimaira 18000
launch mnemo-vllm-jeevy sft7b-jeevy    jeevy    18001
echo "Both oracles launching. Poll :18000 and :18001 /v1/models for readiness (~2min each)."
