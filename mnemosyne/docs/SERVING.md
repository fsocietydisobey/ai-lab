# Serving a codebase-bound model to Claude Code

Goal: Claude Code drives the local model as its engine. Claude Code speaks the
**Anthropic Messages API**; local inference servers speak **OpenAI**. So the stack
is three layers:

```
Claude Code ──Anthropic API──> [shim] ──OpenAI API──> [inference server] ── local model
```

## Step 0 — collapse the model to one artifact

Stack the SFT adapter onto the merged-CPT base so there's a single model to serve:

```bash
python scripts/merge_adapter.py \
    --base models/merged-khimaira-cpt \
    --adapter adapters/sft-khimaira \
    --out models/khimaira-assistant
```

Prove it answers before serving (cheap, no infra):

```bash
python scripts/infer.py --model models/khimaira-assistant \
    --prompt "What does roster_recovery do in khimaira?"
```

## Step 1 — inference server (OpenAI-compatible)

The merged 1.5B in bf16 (~3 GB) serves comfortably on the 8 GB GPU. Two options:

**vLLM** (serves HF safetensors directly — no conversion):
```bash
pip install vllm
vllm serve models/khimaira-assistant --port 8000 --served-model-name khimaira
# NOTE: verify vLLM works on Blackwell/CUDA13 — same bleeding-edge caveat as 4-bit.
```

**Ollama** (needs GGUF conversion via llama.cpp first):
```bash
python llama.cpp/convert_hf_to_gguf.py models/khimaira-assistant --outfile khimaira.gguf
ollama create khimaira -f Modelfile   # Modelfile: FROM ./khimaira.gguf
ollama serve   # exposes OpenAI-compatible /v1 on :11434
```

## Step 2 — Anthropic→OpenAI shim

Claude Code won't talk OpenAI. Bridge with LiteLLM (or a claude-code-proxy):

```bash
pip install 'litellm[proxy]'
# litellm_config.yaml:
#   model_list:
#     - model_name: khimaira
#       litellm_params:
#         model: openai/khimaira
#         api_base: http://localhost:8000/v1
#         api_key: dummy
litellm --config litellm_config.yaml --port 4000
```

LiteLLM exposes an Anthropic-compatible endpoint at `http://localhost:4000`.

## Step 3 — point Claude Code at it

```bash
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_AUTH_TOKEN=dummy
export ANTHROPIC_MODEL=khimaira
claude   # now driving the local khimaira-bound model
```

## Status / honesty

- **Proven:** training pipeline + model quality (`infer.py`).
- **NOT yet run end-to-end:** the serve → shim → Claude Code wiring. It's the
  fiddliest layer with two hardware unknowns (vLLM-on-Blackwell, the Anthropic
  shim), and needs a live interactive run to validate. The commands above are the
  recipe, not a verified deployment.
- **Reality check:** a 1.5B model driving Claude Code will be far weaker at
  reasoning than Opus. The codebase-bound model is best used as a *codebase-knowledge
  tool the orchestrator queries*, or for narrow/offline/private work — not as a
  drop-in Opus replacement. Decide the architecture before investing in serving.
```
