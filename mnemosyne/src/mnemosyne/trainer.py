"""LoRA fine-tune trainer — Phase 2, requires PyTorch + PEFT.

GPU tiers:
  8GB VRAM:   Qwen2.5-7B 4-bit quantized, LoRA adapters
  128GB VRAM: Qwen2.5-72B full fine-tune, drop load_in_4bit

Install before use:
  uv pip install torch transformers peft bitsandbytes accelerate datasets
"""

from __future__ import annotations

from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


def ml_deps_available() -> bool:
    try:
        import torch  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def fine_tune(domain: str, model_name: str, output_dir: Path) -> None:
    if not ml_deps_available():
        raise ImportError(
            "ML deps not installed. Run: uv pip install torch transformers peft bitsandbytes accelerate datasets"
        )

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from mnemosyne import store

    pairs = store.load(domain)
    if not pairs:
        raise ValueError(f"No training data for domain '{domain}'")

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
        load_in_4bit=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
    )

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
        ),
        train_dataset=dataset.map(tokenize, batched=True),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
