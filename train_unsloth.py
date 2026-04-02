import unsloth  # MUST be first
from unsloth import FastLanguageModel
import argparse
import json
import os
os.environ["UNSLOTH_DISABLE_THINKING"] = "1" 
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from PIL import Image

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_json_dataset(json_path: str) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
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

            # ✅ FORCE EVERYTHING TO STRING HERE
            if isinstance(content, list):
                # extract text safely
                text_parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text_parts.append(c.get("text", ""))
                content = " ".join(text_parts)

            if content is None:
                content = ""

            fixed_msgs.append({
                "role": str(role),
                "content": str(content),  # ✅ ALWAYS STRING
            })

        flat.append({
            "messages": fixed_msgs,
            "image_path": str(r["image_path"]),
            "label": str(r.get("label", "")),
        })

    return Dataset.from_list(flat)

def debug_sample(sample, idx=0):
    print(f"\n===== DEBUG SAMPLE {idx} =====")

    print("MESSAGES:")
    for i, m in enumerate(sample["messages"]):
        print(f"  [{i}] role={m['role']}")
        print(f"       type={type(m['content'])}")
        print(f"       content={m['content']}")

    print("\nIMAGE PATH:", sample.get("image_path"))

    print("=============================\n")
    
    
def convert_to_conversation(sample, processor):
    try:
        image = Image.open(sample["image_path"]).convert("RGB")
    except Exception as e:
        print(f"[WARN] Cannot load {sample['image_path']}: {e}")
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    messages = sample["messages"]

    # ✅ ensure roles exist
    assert any(m["role"] == "user" for m in messages), "No user role!"
    assert any(m["role"] == "assistant" for m in messages), "No assistant role!"

    # 🔥 ensure user message is MULTIMODAL STRUCTURED
    for m in messages:
        if m["role"] == "user":

            # already correct → skip
            if isinstance(m["content"], list):
                break

            # convert string → structured
            text = m["content"] if m["content"] is not None else ""

            m["content"] = [
                {"type": "image"},
                {"type": "text", "text": text}
            ]
            break

    # ✅ VALIDATION (UPDATED FOR STRUCTURED FORMAT)
    user_msg = next(m for m in messages if m["role"] == "user")

    assert isinstance(user_msg["content"], list), "User content must be list!"

    # must contain image block
    assert any(
        isinstance(c, dict) and c.get("type") == "image"
        for c in user_msg["content"]
    ), "Missing image block!"

    # must contain text block
    assert any(
        isinstance(c, dict) and c.get("type") == "text"
        for c in user_msg["content"]
    ), "Missing text block!"

    # ✅ ensure image loaded
    assert isinstance(image, Image.Image), "Invalid image!"

    return {
        "messages": messages,
        "images": [image],
    }

def load_model_and_processor(cfg: dict):
    model_name = cfg["model"]["name"]
    lora_cfg = cfg["lora"]

    print(f"Loading model with Unsloth: {model_name}")

    model, processor = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=True,
        dtype=None,  # auto-detect, avoids the float32 warning
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

def debug_converted(sample, idx=0):
    print(f"\n===== CONVERTED SAMPLE {idx} =====")

    print("MESSAGES:")
    for i, m in enumerate(sample["messages"]):
        print(f"  [{i}] role={m['role']}")
        print(f"       content={m['content']}")

    print("\nIMAGE COUNT:", len(sample["images"]))
    print("IMAGE TYPE:", type(sample["images"][0]))

    print("=================================\n")
    
def main():
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

    # ── Switch to training mode (Unsloth-specific, must be before SFTTrainer)
    FastVisionModel.for_training(model)

    # ── Convert datasets to {messages, images} format
    train_dataset = [convert_to_conversation(s, processor) for s in train_dataset]
    debug_converted(train_dataset[0])
    val_dataset   = [convert_to_conversation(s, processor) for s in val_dataset]

    # ── Data collator — this is the key difference from your old version
    data_collator = UnslothVisionDataCollator(
        model,
        processor,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # ── SFTConfig
    output_dir = train_cfg["output_dir_unsloth"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_bf16_supported()

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        optim="adamw_8bit",       # ← Unsloth's optimized optimizer
        bf16=use_bf16,
        fp16=not use_bf16,
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        logging_steps=train_cfg["logging_steps"],
        load_best_model_at_end=False,
        report_to=train_cfg["report_to"],
        dataloader_num_workers=0,  # ← 0 prevents multiprocessing PIL issues
        remove_unused_columns=False,
        # ── Critical: skip TRL's internal dataset processing
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=cfg["model"]["max_seq_length"],
        seed=ds_cfg["seed"],
    )
    
    debug_sample(train_dataset[0])

    # ── SFTTrainer
    import inspect
    trainer_params = inspect.signature(SFTTrainer).parameters
    trainer_kwargs = {
        "model":          model,
        "args":           sft_config,
        "train_dataset":  train_dataset,
        "eval_dataset":   val_dataset,
        "data_collator":  data_collator,
    }
    # TRL 0.12+ renamed tokenizer → processing_class
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
    merge_path = Path(cfg["merge"]["unsloth_merged"])
    merge_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(merge_path), processor, save_method="merged_16bit")
    print(f"✓ Merged model saved: {merge_path}")


if __name__ == "__main__":
    main()