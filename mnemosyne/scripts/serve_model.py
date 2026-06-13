#!/usr/bin/env python3
"""Minimal OpenAI-compatible inference server for a mnemosyne model.

transformers-based (proven on GB10) — not vLLM, which is bleeding-edge on
aarch64/Blackwell. Good enough to serve a single-user codebase oracle and to
prove the Claude Code → shim → OpenAI → local-model loop. Swap for vLLM later.

  MNEMO_MODEL=models/sft7b-khimaira python scripts/serve_model.py --port 18000

Endpoints:
  GET  /health
  POST /v1/chat/completions   (OpenAI-compatible)
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MNEMO_MODEL", "models/sft7b-khimaira")
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

app = FastAPI(title="mnemosyne")
print(f"[serve] loading {MODEL} (this takes ~2 min for a 7B) ...", flush=True)
_tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=MODELS_DIR)
if _tok.pad_token is None:
    _tok.pad_token = _tok.eos_token
_model = AutoModelForCausalLM.from_pretrained(
    MODEL, cache_dir=MODELS_DIR, torch_dtype=torch.bfloat16, device_map="auto"
)
_model.eval()
print("[serve] model ready", flush=True)


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    model: str = "khimaira"
    messages: list[Msg]
    max_tokens: int = 256
    temperature: float = 0.7


def _to_prompt(messages: list[Msg]) -> str:
    # Match the SFT training format exactly: ### Instruction / ### Response.
    user = "\n".join(m.content for m in messages if m.role in ("user", "system"))
    return f"### Instruction:\n{user}\n\n### Response:\n"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL,
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


@app.post("/v1/chat/completions")
def chat(req: ChatReq) -> dict:
    prompt = _to_prompt(req.messages)
    ids = _tok(prompt, return_tensors="pt").to(_model.device)
    t0 = time.time()
    with torch.no_grad():
        out = _model.generate(
            **ids, max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0,
            temperature=max(req.temperature, 1e-5), top_p=0.9,
            pad_token_id=_tok.pad_token_id or _tok.eos_token_id,
        )
    gen = out[0][ids["input_ids"].shape[1]:]
    text = _tok.decode(gen, skip_special_tokens=True).strip()
    # the SFT'd model sometimes re-emits the format — cut at the next turn.
    text = text.split("### Instruction")[0].strip()
    n_in, n_out = int(ids["input_ids"].shape[1]), int(gen.shape[0])
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                  "total_tokens": n_in + n_out, "latency_s": round(time.time() - t0, 2)},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
