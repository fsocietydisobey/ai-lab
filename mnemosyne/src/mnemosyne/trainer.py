"""LoRA fine-tune trainer — Phase 2 (requires PyTorch + PEFT).

Phase 1 uses the querier with base model + injected context (no training required).
Phase 2 runs LoRA delta updates after each session batch.

GPU tiers supported:
  laptop (8GB VRAM):  Qwen2.5-7B-Instruct at 4-bit — LoRA adapters only
  DGX Spark (128GB):  Qwen2.5-72B-Instruct full fine-tune, parallel domains

Install ML deps before using this module:
  uv pip install torch transformers peft bitsandbytes accelerate datasets
"""

from __future__ import annotations

from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


def check_deps() -> bool:
    """Return True if ML deps are installed."""
    try:
        import torch  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def fine_tune(domain: str, model_name: str, output_dir: Path) -> None:
    """Run a LoRA delta fine-tune for a domain using accumulated training data.

    Requires ML deps (torch, transformers, peft, bitsandbytes, accelerate).
    Call check_deps() first — raises ImportError if deps missing.
    """
    if not check_deps():
        raise ImportError(
            "ML deps not installed. Run: uv pip install torch transformers peft bitsandbytes accelerate datasets"
        )

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    import torch

    from mnemosyne import store

    pairs = store.load(domain)
    if not pairs:
        raise ValueError(f"No training data for domain '{domain}'")

    # Format as instruction-response pairs
    texts = [
        f"### Instruction:\n{p['instruction']}\n\n### Response:\n{p['response']}"
        for p in pairs
    ]
    dataset = Dataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODELS_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=MODELS_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,  # 4-bit for laptop GPU; remove for DGX full fine-tune
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
