"""
train_unsloth.py
================
Fine-tune Qwen3.5-0.8B with QLoRA using Unsloth FastVisionModel.
~3GB VRAM, 1.5-2x faster than TRL, same results.

Install:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

Usage:
  python train_unsloth.py
  python train_unsloth.py --config config.yaml --resume
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from PIL import Image
from transformers import TrainingArguments,Trainer

try:
    from unsloth import FastVisionModel
    from trl import SFTTrainer
except ImportError:
    print("[ERROR] Unsloth not installed. Run:")
    print('  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
    import sys; sys.exit(1)


# ── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Dataset ──────────────────────────────────────────────────────────────────

def load_json_dataset(json_path: str) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def make_hf_dataset(records):
    return Dataset.from_list([
        {
            "messages_json": json.dumps(r["messages"], ensure_ascii=False),
            "image_path": r["image_path"],
            "label": r.get("label", ""),
        }
        for r in records
    ])


# ── Collator ─────────────────────────────────────────────────────────────────

class VLMDataCollator:
    def __init__(self, processor, max_seq_length: int = 1024):
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __call__(self, batch: list[dict]) -> dict:
        texts, images = [], []

        for sample in batch:
            try:
                image = Image.open(sample["image_path"]).convert("RGB")
            except Exception as e:
                print(f"[WARN] Cannot load image {sample['image_path']}: {e}")
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))
            images.append(image)

            messages = json.loads(sample["messages_json"])
            text = self.processor.apply_chat_template(
                messages,
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


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model_and_processor(cfg: dict):
    model_name = cfg["model"]["name"]
    lora_cfg = cfg["lora"]

    # Use Unsloth's pre-quantized version if available, fall back to HF name
    # unsloth/ prefix gets Unsloth's optimized 4-bit weights directly
    unsloth_name = model_name.replace("Qwen/", "unsloth/")
    print(f"Loading model with Unsloth: {unsloth_name}")

    model, processor = FastVisionModel.from_pretrained(
        model_name=unsloth_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
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


class UnslothVLMTrainer(Trainer):
    """
    Subclass that unpacks our collator's dict and explicitly calls
    model(**inputs) to get the loss — bypassing Unsloth's patched
    _unsloth_training_step which expects SFTTrainer's data layout.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)

        # outputs is a CausalLMOutputWithPast; .loss is the CE loss tensor
        loss = outputs.loss

        if loss is None:
            # Fallback: compute manually (shouldn't happen with labels set)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return (loss, outputs) if return_outputs else loss
    
# ── Training ─────────────────────────────────────────────────────────────────

def main():
    # Single GPU — same reason as TRL version
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    ds_cfg = cfg["dataset"]

    # ── Load data
    print("Loading datasets ...")
    train_records = load_json_dataset(ds_cfg["train_json"])
    val_records   = load_json_dataset(ds_cfg["val_json"])
    print(f"  Train: {len(train_records)}  |  Val: {len(val_records)}")

    train_dataset = make_hf_dataset(train_records)
    val_dataset   = make_hf_dataset(val_records)

    # ── Load model
    model, processor = load_model_and_processor(cfg)

    # Switch to training mode (Unsloth-specific — must be called before Trainer)
    FastVisionModel.for_training(model)

    # ── Collator
    collator = VLMDataCollator(processor, max_seq_length=cfg["model"]["max_seq_length"])

    # ── TrainingArguments — mirrors TRL version exactly
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
        load_best_model_at_end=False,
        report_to=train_cfg["report_to"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        local_rank=-1,
        ddp_find_unused_parameters=False,
    )

    # ── Trainer — use base Trainer like TRL version (not SFTTrainer)
    # SFTTrainer tries to reprocess text which breaks our custom collator
    from transformers import Trainer
    trainer = UnslothVLMTrainer(
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

    # ── Save adapter
    final_path = Path(output_dir) / "final_adapter"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n✓ Adapter saved: {final_path}")

    # ── Optional: merge to fp16 for deployment
    merge_path = Path(cfg["merge"]["unsloth_merged"])
    merge_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(merge_path), processor, save_method="merged_16bit")
    print(f"✓ Merged model saved: {merge_path}")


if __name__ == "__main__":
    main()