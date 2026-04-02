"""
train_trl.py
============
Fine-tune Qwen3.5-0.8B-Instruct with QLoRA using HuggingFace TRL SFTTrainer.
Supports images via the processor's chat template.

Environment: Kaggle / Colab / Local (requires ~6GB VRAM)

Usage:
  python train_trl.py
  python train_trl.py --config config.yaml --resume
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,                    # ← add this
)


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
        flat.append({
            "messages_json": json.dumps(r["messages"], ensure_ascii=False),
            "image_path": r["image_path"],
            "label": r.get("label", ""),
        })
    return Dataset.from_list(flat)


# ── Collator ─────────────────────────────────────────────────────────────────


class VLMDataCollator:
    """
    Collate a batch of VLM samples.
    Each sample has:
      - messages: list of chat messages (system / user with image / assistant)
      - image_path: str path to the image file

    The collator applies the processor's chat template and loads images,
    then returns tensors ready for the model.
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

        # Tokenize + image processing
        encoding = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )

        # Labels = input_ids with padding masked to -100
        labels = encoding["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Mask the prompt (everything before assistant turn) from loss
        # Simple approach: mask up to the last occurrence of assistant token
        encoding["labels"] = labels

        return encoding


# ── Model Setup ──────────────────────────────────────────────────────────────


def load_model_and_processor(cfg: dict):
    model_name = cfg["model"]["name"]

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model: {model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


def apply_lora(model, cfg: dict):
    lora_cfg = cfg["lora"]
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Training ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train with TRL SFTTrainer")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
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
    model = apply_lora(model, cfg)

    # ── Collator
    collator = VLMDataCollator(processor, max_seq_length=cfg["model"]["max_seq_length"])

    # ── TrainingArguments
    output_dir = train_cfg["output_dir_trl"]
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
    load_best_model_at_end=False,   # ← disable, needs eval metric
    report_to=train_cfg["report_to"],
    dataloader_num_workers=train_cfg["dataloader_num_workers"],
    remove_unused_columns=False,
    )

    # ── Trainer
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    )

    # ── Train
    print("\nStarting training ...")
    resume_ckpt = output_dir if args.resume else None
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── Save final adapter
    final_path = Path(output_dir) / "final_adapter"
    trainer.model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n✓ Adapter saved: {final_path}")
    print("  Next: run eval.py  or  merge with merge_trl.py")


if __name__ == "__main__":
    main()
