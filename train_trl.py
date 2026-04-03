"""
train_trl.py
============
Fine-tune Qwen3.5-0.8B-Instruct with QLoRA using HuggingFace TRL SFTTrainer.
Supports images via the processor's chat template.

Environment: Kaggle / Colab / Local (requires ~6GB VRAM)

Usage:
  python train_trl.py                  ← full training, debug off
  python train_trl.py debug=on         ← smoke test + verbose debug
  python train_trl.py --resume         ← resume full training
"""

import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3_5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)


# ── Config ───────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge: override takes priority over base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result


# ── Debug helpers ─────────────────────────────────────────────────────────────


def dbg_sep(title: str):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def debug_raw_sample(sample, idx: int, dbg: dict):
    """Show raw HF dataset row (messages stored as JSON string)."""
    if not (dbg.get("enabled") and dbg.get("show_raw")):
        return
    dbg_sep(f"RAW SAMPLE [{idx}]  (HF dataset row)")
    print(f"  image_path : {sample.get('image_path')}")
    print(f"  label      : {sample.get('label')}")
    messages = json.loads(sample["messages_json"])
    print("  messages   :")
    for i, m in enumerate(messages):
        content = m["content"]
        print(f"    [{i}] role={m['role']}")
        if isinstance(content, list):
            for block in content:
                btype = block.get("type")
                if btype == "image":
                    print(f"         [image block]")
                elif btype == "text":
                    preview = block.get("text", "")[:120].replace("\n", "↵")
                    print(f"         [text] {preview!r}")
        else:
            preview = str(content)[:120].replace("\n", "↵")
            print(f"         {preview!r}{'...' if len(str(content)) > 120 else ''}")


def debug_tokenized_sample(sample, idx: int, processor, max_seq_length: int, dbg: dict):
    """Show token count and whether the sample fits in context."""
    if not (dbg.get("enabled") and dbg.get("show_tokenized")):
        return
    dbg_sep(f"TOKENIZED SAMPLE [{idx}]")
    try:
        messages = json.loads(sample["messages_json"])
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        ids = processor.tokenizer(text, truncation=False)["input_ids"]
        fits = len(ids) <= max_seq_length
        print(f"  token count  : {len(ids)}")
        print(f"  max_seq_len  : {max_seq_length}")
        print(f"  fits context : {'✓ YES' if fits else '✗ NO — will be truncated'}")
        print(f"  prompt preview (first 200 chars):")
        print(f"    {text[:200].replace(chr(10), '↵')!r}")
    except Exception as e:
        print(f"  [WARN] tokenization failed: {e}")


def debug_label_distribution(dataset, dbg: dict):
    """Show SAFE/UNSAFE counts."""
    if not dbg.get("enabled"):
        return
    dbg_sep("LABEL DISTRIBUTION")
    from collections import Counter

    labels = [s.get("label", "UNKNOWN") for s in dataset]
    counts = Counter(labels)
    total = len(labels)
    for lbl, cnt in sorted(counts.items()):
        print(f"  {lbl:<10} {cnt:>6}  ({cnt / total * 100:.1f}%)")


def debug_gpu(dbg: dict):
    """Show per-GPU memory stats."""
    if not (dbg.get("enabled") and dbg.get("show_gpu")):
        return
    dbg_sep("GPU MEMORY")
    if not torch.cuda.is_available():
        print("  No CUDA device found.")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        free = total - reserved
        print(f"  GPU {i}: {props.name}")
        print(f"    Total     : {total:.2f} GB")
        print(f"    Reserved  : {reserved:.2f} GB")
        print(f"    Allocated : {allocated:.2f} GB")
        print(f"    Free      : {free:.2f} GB")


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


class VLMDataCollator:
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


# ── Model ────────────────────────────────────────────────────────────────────


def load_model_and_processor(cfg: dict):
    model_name = cfg["model"]["name"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model: {model_name}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    inner = model.model
    for attr in ("visual", "vision_model", "vision_encoder"):
        if hasattr(inner, attr):
            setattr(inner, attr, getattr(inner, attr).to(torch.float16))
            print(f"  Cast {attr} to float16")
            break

    visual_module = model.model.visual
    visual_cls = type(visual_module)
    if not getattr(visual_cls, "_dtype_patched", False):

        def _patched_dtype(self):
            try:
                return next(p.dtype for p in self.parameters() if p.is_floating_point())
            except StopIteration:
                return torch.float16

        visual_cls.dtype = property(_patched_dtype)
        visual_cls._dtype_patched = True

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


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description="Train with TRL SFTTrainer")
    parser.add_argument("--config", default="config/config_trl.yaml")
    parser.add_argument("--base_config", default="config/config_base.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("extra", nargs="*", help="Extra flags e.g. debug=on")
    args = parser.parse_args()

    # Parse extra flags:  debug=on / debug=off
    extra_flags = {k: v for k, v in (f.split("=", 1) for f in args.extra if "=" in f)}
    cli_debug = extra_flags.get("debug", "").lower()

    # ── Load and merge configs
    base_cfg = load_config(args.base_config)
    trl_cfg = load_config(args.config)
    cfg = merge_configs(base_cfg, trl_cfg)

    train_cfg = cfg["training"]
    ds_cfg = cfg["dataset"]
    smoke_cfg = cfg.get("smoke_test", {})

    # CLI flag overrides config file
    dbg = cfg.get("debug", {"enabled": False})
    if cli_debug == "on":
        dbg["enabled"] = True
    elif cli_debug == "off":
        dbg["enabled"] = False

    smoke_mode = dbg.get("enabled", False)

    if smoke_mode:
        print("\n" + "=" * 60)
        print("  SMOKE TEST / DEBUG MODE ON")
        print("  Running minimal steps to verify full pipeline.")
        print("  To disable: python train_trl.py  (no flag)")
        print("=" * 60)
    else:
        print("\n[DEBUG OFF] Full training run. Pass debug=on to enable smoke test.")

    # ── Load data
    print("\nLoading datasets ...")
    train_records = load_json_dataset(ds_cfg["train_json"])
    val_records = load_json_dataset(ds_cfg["val_json"])

    if smoke_mode:
        n = smoke_cfg.get("max_samples", 32)
        train_records = train_records[:n]
        val_records = val_records[:n]
        print(f"  [SMOKE] Using {n} train + {n} val samples")

    print(f"  Train: {len(train_records)}  |  Val: {len(val_records)}")

    train_dataset = make_hf_dataset(train_records)
    val_dataset = make_hf_dataset(val_records)

    # ── Debug: raw samples
    n_dbg = dbg.get("n_samples", 2)
    for i in range(min(n_dbg, len(train_dataset))):
        debug_raw_sample(train_dataset[i], idx=i, dbg=dbg)

    # ── Debug: label distribution
    debug_label_distribution(train_records, dbg=dbg)

    # ── Load model
    model, processor = load_model_and_processor(cfg)
    model = apply_lora(model, cfg)

    # ── Debug: tokenized samples (needs processor)
    max_seq = cfg["model"]["max_seq_length"]
    for i in range(min(n_dbg, len(train_dataset))):
        debug_tokenized_sample(
            train_dataset[i],
            idx=i,
            processor=processor,
            max_seq_length=max_seq,
            dbg=dbg,
        )

    # ── Debug: GPU memory
    debug_gpu(dbg=dbg)

    # ── Collator
    collator = VLMDataCollator(processor, max_seq_length=max_seq)

    # ── TrainingArguments — smoke overrides applied here
    output_dir = train_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    num_epochs = (
        smoke_cfg.get("num_train_epochs", train_cfg["num_train_epochs"])
        if smoke_mode
        else train_cfg["num_train_epochs"]
    )
    max_steps = smoke_cfg.get("max_steps", -1) if smoke_mode else -1
    save_steps = (
        smoke_cfg.get("save_steps", train_cfg["save_steps"])
        if smoke_mode
        else train_cfg["save_steps"]
    )
    eval_steps = (
        smoke_cfg.get("eval_steps", train_cfg["eval_steps"])
        if smoke_mode
        else train_cfg["eval_steps"]
    )
    logging_steps = (
        smoke_cfg.get("logging_steps", train_cfg["logging_steps"])
        if smoke_mode
        else train_cfg["logging_steps"]
    )

    if smoke_mode:
        print(
            f"\n  [SMOKE] epochs={num_epochs}  max_steps={max_steps}"
            f"  save_steps={save_steps}  eval_steps={eval_steps}"
        )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
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
        save_steps=save_steps,
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        load_best_model_at_end=False,
        report_to=train_cfg["report_to"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        local_rank=-1,
        ddp_find_unused_parameters=False,
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

    if smoke_mode:
        print("\n" + "=" * 60)
        print("  ✓ SMOKE TEST PASSED — full pipeline completed.")
        print("  Verified: load → train → save adapter")
        print("  Run without debug=on for full training.")
        print("=" * 60)
    else:
        print("  Next: run eval.py  or  merge with merge_trl.py")


if __name__ == "__main__":
    main()
