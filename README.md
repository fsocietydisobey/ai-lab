# ai-lab

Local AI infrastructure for khimaira. Runs as sidecars alongside the khimaira daemon.

## Services

| Service | Port | Purpose |
|---|---|---|
| mnemosyne | 8765 | Persistent domain memory — LoRA fine-tune + query |

## Models

Shared model weights live in `models/`. Downloaded once, used by all services.

| Model | Size | Used by |
|---|---|---|
| qwen2.5-7b-instruct | ~15GB | mnemosyne |

## Setup

Each service has its own isolated venv:

```bash
cd mnemosyne
uv venv --python 3.12
uv pip install -e ".[cuda]"   # installs PyTorch with CUDA
mnemosyne serve
```

## Architecture

khimaira daemon → HTTP → mnemosyne service

Lead sessions call mnemosyne at boot to load domain memory. Sessions write distilled
learnings at end. Periodic LoRA fine-tune runs update the model weights.
