"""
train_unsloth.py
================
Fine-tune Qwen3.5-0.8B-Instruct with QLoRA using Unsloth.
~3GB VRAM, 2-2.7x faster than TRL baseline, identical results.

Install Unsloth FIRST (env-specific):
  Kaggle/Colab CUDA 12.1:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  Check: https://unsloth.ai/docs/get-started/installation

Usage:
  python train_unsloth.py
  python train_unsloth.py --config config.yaml --resume
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from datasets import Dataset
from transformers import TrainingArguments

# Unsloth imports — will raise ImportError if not installed
try:
    from unsloth import FastVisionModel
    from trl import SFTTrainer

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("[ERROR] Unsloth not installed.")
    print("Install with:")
    print(
        '  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
    )
    import sys

    sys.exit(1)


# ── Config ──────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Dataset ──────────────────────────────────────────────────────────────────


def load_json_dataset(json_path: str) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def make_hf_dataset(records):
    flat = []
    for r in records:
        flat.append(
            {
                "messages_json": json.dumps(r["messages"], ensure_ascii=False),
                "image_path": r["image_path"],
                "label": r.get("label", ""),
            }
        )
    return Dataset.from_list(flat)


# ── Collator ─────────────────────────────────────────────────────────────────


class UnslothVLMCollator:
    """
    Same collator logic as TRL version, adapted for Unsloth processor API.
    """

    def __init__(self, processor, max_seq_length: int = 2048):
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __call__(self, batch: list[dict]) -> dict:
        texts = []
        images = []

        for sample in batch:
            img_path = sample["image_path"]
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Cannot load image {img_path}: {e}")
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))

            images.append(image)

            # Deserialize messages back from JSON string
            messages = json.loads(sample["messages_json"])  # ← add this

            text = self.processor.apply_chat_template(
                messages,  # ← use messages, not sample["messages"]
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        encoding = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )

        labels = encoding["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoding["labels"] = labels

        return encoding


# ── Model ────────────────────────────────────────────────────────────────────


def load_model_and_processor(cfg: dict):
    model_name = cfg["model"]["name"]
    lora_cfg = cfg["lora"]

    print(f"Loading model with Unsloth: {model_name}")

    model, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",  # key optimization
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,  # fine-tune vision encoder too
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        random_state=42,
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


# ── Training ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train with Unsloth (optimized)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    ds_cfg = cfg["dataset"]

    # ── Load data
    print("Loading datasets ...")
    train_records = load_json_dataset(ds_cfg["train_json"])
    val_records = load_json_dataset(ds_cfg["val_json"])
    print(f"  Train: {len(train_records)}  |  Val: {len(val_records)}")

    train_dataset = make_hf_dataset(train_records)
    val_dataset = make_hf_dataset(val_records)

    # ── Load model
    model, processor = load_model_and_processor(cfg)

    # Enable for inference (Unsloth-specific)
    FastVisionModel.for_training(model)

    # ── Collator
    collator = UnslothVLMCollator(processor, cfg["model"]["max_seq_length"])

    # ── Training args (same as TRL, different output dir)
    output_dir = train_cfg["output_dir_unsloth"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        optim=train_cfg["optim"],
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        logging_steps=train_cfg["logging_steps"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        report_to=train_cfg["report_to"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=False,
    )

    # ── Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # ── Train
    print("\nStarting Unsloth training ...")
    resume_ckpt = output_dir if args.resume else None
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── Save
    final_path = Path(output_dir) / "final_adapter"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n✓ Adapter saved: {final_path}")

    # Optionally merge + save in float16 for deployment
    merge_path = Path(cfg["merge"]["unsloth_merged"])
    merge_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(merge_path), processor, save_method="merged_16bit")
    print(f"✓ Merged model saved: {merge_path}")


if __name__ == "__main__":
    main()
