# ai-lab

Local AI infrastructure for khimaira. Runs as sidecars alongside the khimaira daemon.

## Services

| Service | Port | Purpose |
|---|---|---|
| mnemosyne | 8766 | Persistent domain memory — distill + query + LoRA fine-tune |

## Models

Shared model weights live in `models/`. Downloaded once, used by all services.

| Model | Size | GPU tier |
|---|---|---|
| Qwen2.5-7B-Instruct (4-bit) | ~4.5GB VRAM | Laptop RTX Pro 1000 (8GB) |
| Qwen2.5-72B-Instruct (full) | ~128GB VRAM | DGX Spark (128GB) |

---

## Architecture

### Phase 1 — Distill + Query (no GPU required)

```mermaid
sequenceDiagram
    participant Lead as Domain Lead Session
    participant K as Khimaira Daemon
    participant M as Mnemosyne :8766
    participant C as Claude API (Haiku)
    participant S as JSONL Store

    Lead->>K: session ends
    K->>M: POST /distill {domain, transcript, session_slug}
    M->>C: extract training pairs from transcript
    C-->>M: [{instruction, response}, ...]
    M->>S: append to data/{domain}.jsonl
    M-->>K: {pairs_extracted, total_pairs}

    Note over Lead,S: Next session boots

    Lead->>M: POST /query {domain, question}
    M->>S: load recent pairs (top 20)
    M->>C: answer using pairs as context
    C-->>M: domain-specific answer
    M-->>Lead: answer
```

### Phase 2 — LoRA Fine-Tune (GPU)

```mermaid
flowchart LR
    subgraph laptop["Laptop — RTX Pro 1000 (8GB)"]
        S[(JSONL Store)]
        API[Mnemosyne API]
        Inf[Inference\nQwen2.5-7B 4-bit]
    end

    subgraph dgx["DGX Spark (128GB VRAM)"]
        Train[LoRA Trainer\nQwen2.5-72B full]
        Weights[(Model Weights)]
    end

    S -->|training pairs| Train
    Train -->|updated adapter| Weights
    Weights -->|serve| Inf
    API -->|query| Inf
    API -->|distill| S
```

### Khimaira Integration

```mermaid
flowchart TD
    User([Joseph]) --> Intake
    Intake --> Master
    Master --> Lead[Domain Lead\nbackend / data / devops]

    Lead -->|session end hook| Distill[POST /distill]
    Lead -->|session boot| Query[POST /query]

    subgraph mnemosyne["Mnemosyne :8766"]
        Distill --> Store[(domain.jsonl)]
        Query --> Store
        Store --> Model[Base Model\nor LoRA Adapter]
        Model --> Query
    end

    Query -->|domain memory| Lead
```

---

## Setup

```bash
cd mnemosyne
uv venv --python 3.12
uv pip install -e .
mnemosyne serve          # starts at 127.0.0.1:8766
```

For Phase 2 (LoRA fine-tuning):

```bash
uv pip install torch transformers peft bitsandbytes accelerate datasets
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service health + domain list |
| GET | `/domains` | Training pair counts per domain |
| POST | `/query` | Answer a domain memory question |
| POST | `/distill` | Extract training pairs from a session transcript |
