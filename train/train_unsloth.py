import unsloth  # MUST be first
from unsloth import FastLanguageModel
import argparse
import json
import os

os.environ["UNSLOTH_DISABLE_THINKING"] = "1"
from pathlib import Path

# ── Project root resolution ───────────────────────────────────────────────────
# This file lives at <ROOT>/train/train_unsloth.py → ROOT is two levels up
ROOT = Path(__file__).resolve().parent.parent

import torch
import yaml
from datasets import Dataset
from PIL import Image

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

"""
train_unsloth.py
================
Fine-tune Qwen3.5-0.8B with Unsloth (~3GB VRAM, 2-2.7x faster).

Usage:
  python train/train_unsloth.py
  python train/train_unsloth.py --config config/config_unsloth.yaml --base_config config/config_base.yaml --resume

Debug on/off: set debug.enabled in config/config_unsloth.yaml
"""


# ── Config ───────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(ROOT / path) as f:
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


def resolve(path: str) -> Path:
    """Resolve a config path string relative to project ROOT."""
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


# ── Debug helpers ─────────────────────────────────────────────────────────────


def dbg_sep(title: str):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def debug_raw_sample(sample, idx: int, dbg: dict):
    """Show the raw HF dataset row before conversion."""
    if not (dbg.get("enabled") and dbg.get("show_raw")):
        return
    dbg_sep(f"RAW SAMPLE [{idx}]  (pre-conversion)")
    print(f"  image_path : {sample.get('image_path')}")
    print(f"  label      : {sample.get('label')}")
    print("  messages   :")
    for i, m in enumerate(sample["messages"]):
        content = m["content"]
        print(f"    [{i}] role={m['role']}")
        print(f"         type={type(content).__name__}")
        if isinstance(content, str):
            preview = content[:120].replace("\n", "↵")
            print(f"         content={preview!r}{'...' if len(content) > 120 else ''}")
        else:
            print(f"         content={content}")


def debug_converted_sample(sample, idx: int, dbg: dict):
    """Show the converted {messages, images} dict."""
    if not (dbg.get("enabled") and dbg.get("show_converted")):
        return
    dbg_sep(f"CONVERTED SAMPLE [{idx}]  (post-conversion)")
    print(f"  image count : {len(sample['images'])}")
    img = sample["images"][0]
    print(f"  image size  : {img.size}  mode={img.mode}")
    print("  messages    :")
    for i, m in enumerate(sample["messages"]):
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
    """Show token count and whether sample fits in context."""
    if not (dbg.get("enabled") and dbg.get("show_tokenized")):
        return
    dbg_sep(f"TOKENIZED SAMPLE [{idx}]")
    try:
        text = processor.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        )
        ids = processor.tokenizer(text, truncation=False)["input_ids"]
        fits = len(ids) <= max_seq_length
        print(f"  token count  : {len(ids)}")
        print(f"  max_seq_len  : {max_seq_length}")
        print(f"  fits context : {'✓ YES' if fits else '✗ NO — will be filtered'}")
        print(f"  prompt preview (first 200 chars):")
        print(f"    {text[:200].replace(chr(10), '↵')!r}")
    except Exception as e:
        print(f"  [WARN] tokenization failed: {e}")


def debug_gpu(dbg: dict):
    """Show GPU memory stats."""
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


def debug_label_distribution(dataset, dbg: dict):
    """Show label counts in a dataset list."""
    if not dbg.get("enabled"):
        return
    dbg_sep("LABEL DISTRIBUTION")
    from collections import Counter
    import re

    labels = []
    for s in dataset:
        for m in s["messages"]:
            if m["role"] == "assistant":
                content = m["content"] if isinstance(m["content"], str) else ""
                match = re.search(r"LABEL\s*:\s*(SAFE|UNSAFE)", content, re.IGNORECASE)
                labels.append(match.group(1).upper() if match else "UNKNOWN")
                break
    counts = Counter(labels)
    total = len(labels)
    for lbl, cnt in sorted(counts.items()):
        print(f"  {lbl:<10} {cnt:>6}  ({cnt / total * 100:.1f}%)")


def debug_filter_summary(before: int, after: int, max_len: int, dbg: dict):
    """Show how many samples were filtered."""
    if not dbg.get("enabled"):
        return
    removed = before - after
    dbg_sep("CONTEXT FILTER SUMMARY")
    print(f"  Max seq length : {max_len}")
    print(f"  Before filter  : {before}")
    print(f"  After filter   : {after}")
    print(f"  Removed        : {removed}  ({removed / before * 100:.1f}%)")


# ── Dataset ──────────────────────────────────────────────────────────────────


def load_json_dataset(json_path) -> list[dict]:
    with open(resolve(json_path), encoding="utf-8") as f:
        return json.load(f)


def make_hf_dataset(records):
    flat = []
    for r in records:
        msgs = r["messages"]

        if not isinstance(msgs, list):
            msgs = [{"role": "user", "content": str(msgs)}]

        fixed_msgs = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")

            if isinstance(content, list):
                text_parts = [
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                ]
                content = " ".join(text_parts)

            if content is None:
                content = ""

            fixed_msgs.append({"role": str(role), "content": str(content)})

        flat.append(
            {
                "messages": fixed_msgs,
                "image_path": str(r["image_path"]),
                "label": str(r.get("label", "")),
            }
        )
    return Dataset.from_list(flat)


# ── Conversion ───────────────────────────────────────────────────────────────


def convert_to_conversation(sample, processor):
    try:
        image = Image.open(sample["image_path"]).convert("RGB")
    except Exception as e:
        print(f"[WARN] Cannot load {sample['image_path']}: {e}")
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    messages = sample["messages"]

    assert any(m["role"] == "user" for m in messages), "No user role!"
    assert any(m["role"] == "assistant" for m in messages), "No assistant role!"

    for m in messages:
        if m["role"] == "user":
            if isinstance(m["content"], list):
                break
            text = m["content"] if m["content"] is not None else ""
            m["content"] = [{"type": "image"}, {"type": "text", "text": text}]
            break

    user_msg = next(m for m in messages if m["role"] == "user")
    assert isinstance(user_msg["content"], list), "User content must be list!"
    assert any(
        isinstance(c, dict) and c.get("type") == "image" for c in user_msg["content"]
    ), "Missing image block!"
    assert any(
        isinstance(c, dict) and c.get("type") == "text" for c in user_msg["content"]
    ), "Missing text block!"
    assert isinstance(image, Image.Image), "Invalid image!"

    return {"messages": messages, "images": [image]}


# ── Context filter ───────────────────────────────────────────────────────────


def fits_in_context(sample, processor, max_seq_length: int) -> bool:
    try:
        text = processor.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        )
        ids = processor.tokenizer(text, truncation=False)["input_ids"]
        return len(ids) <= max_seq_length
    except Exception:
        return False


# ── Model ────────────────────────────────────────────────────────────────────


def load_model_and_processor(cfg: dict):
    model_name = cfg["model"]["name"]
    lora_cfg = cfg["lora"]

    print(f"Loading model with Unsloth: {model_name}")

    model, processor = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_unsloth.yaml")
    parser.add_argument("--base_config", default="config/config_base.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("extra", nargs="*", help="Extra flags e.g. debug=on")
    args = parser.parse_args()

    # Parse extra flags like debug=on / debug=off
    extra_flags = {k: v for k, v in (f.split("=", 1) for f in args.extra if "=" in f)}
    cli_debug = extra_flags.get("debug", "").lower()

    # ── Load and merge configs
    base_cfg = load_config(args.base_config)
    unsloth_cfg = load_config(args.config)
    cfg = merge_configs(base_cfg, unsloth_cfg)

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
        print("  To disable: python train/train_unsloth.py  (no flag)")
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

    # ── Load model
    model, processor = load_model_and_processor(cfg)

    # ── Switch to training mode
    FastVisionModel.for_training(model)

    # ── Convert datasets
    train_dataset = [convert_to_conversation(s, processor) for s in train_dataset]
    val_dataset = [convert_to_conversation(s, processor) for s in val_dataset]

    # ── Debug: converted samples
    for i in range(min(n_dbg, len(train_dataset))):
        debug_converted_sample(train_dataset[i], idx=i, dbg=dbg)

    # ── Debug: tokenized samples
    max_len = cfg["model"]["max_seq_length"]
    for i in range(min(n_dbg, len(train_dataset))):
        debug_tokenized_sample(
            train_dataset[i],
            idx=i,
            processor=processor,
            max_seq_length=max_len,
            dbg=dbg,
        )

    # ── Debug: label distribution
    debug_label_distribution(train_dataset, dbg=dbg)

    # ── Filter context length
    before = len(train_dataset)
    train_dataset = [s for s in train_dataset if fits_in_context(s, processor, max_len)]
    val_dataset = [s for s in val_dataset if fits_in_context(s, processor, max_len)]
    debug_filter_summary(before, len(train_dataset), max_len, dbg=dbg)

    # ── Debug: GPU memory
    debug_gpu(dbg=dbg)

    # ── Data collator
    data_collator = UnslothVisionDataCollator(
        model,
        processor,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # ── SFTConfig
    output_dir = str(resolve(train_cfg["output_dir"]))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_bf16_supported()

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

    sft_config = SFTConfig(
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
        optim="adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        save_strategy=train_cfg["save_strategy"],
        save_steps=save_steps,
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        load_best_model_at_end=False,
        report_to=train_cfg["report_to"],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=max_len,
        seed=ds_cfg["seed"],
    )

    # ── SFTTrainer
    import inspect

    trainer_params = inspect.signature(SFTTrainer).parameters
    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor

    trainer = SFTTrainer(**trainer_kwargs)

    # ── Train
    print("\nStarting Unsloth training ...")
    resume_ckpt = output_dir if args.resume else None
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── Save adapter
    final_path = Path(output_dir) / "final_adapter"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n✓ Adapter saved: {final_path}")

    # ── Merge to fp16
    merge_path = resolve(cfg["merge"]["merged"])
    merge_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(merge_path), processor, save_method="merged_16bit")
    print(f"✓ Merged model saved: {merge_path}")

    if smoke_mode:
        print("\n" + "=" * 60)
        print("  ✓ SMOKE TEST PASSED — full pipeline completed.")
        print("  Verified: load → train → save adapter → merge")
        print("  Run without debug=on for full training.")
        print("=" * 60)


if __name__ == "__main__":
    main()
