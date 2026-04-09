"""
train_trl.py
============
Fine-tune Qwen3.5-0.8B-Instruct with QLoRA using HuggingFace TRL SFTTrainer.

Changes vs. original:
  - Uses AutoModelForImageTextToText (not Qwen3_5ForConditionalGeneration)
  - Uses torch.float32 throughout (not float16) for numerical stability
  - VLMDataCollator handles both:
      * regular image samples   (image_path ends with .jpg/.png/etc.)
      * video tensor samples    (image_path ends with .pt)
        → loads pixel_values_videos + video_grid_thw from the .pt file
          produced by 00_extract_frames.py

Environment: Kaggle / Colab / Local (requires ~6 GB VRAM minimum)

Usage:
  python train/train_trl.py                  ← full training
  python train/train_trl.py debug=on         ← smoke test + verbose debug
  python train/train_trl.py --resume         ← resume from last checkpoint
"""

import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)


# ── Config ────────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ── Debug helpers ─────────────────────────────────────────────────────────────


def dbg_sep(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def debug_raw_sample(sample, idx: int, dbg: dict):
    if not (dbg.get("enabled") and dbg.get("show_raw")):
        return
    dbg_sep(f"RAW SAMPLE [{idx}]")
    print(f"  image_path : {sample.get('image_path')}")
    print(f"  label      : {sample.get('label')}")
    messages = json.loads(sample["messages_json"])
    for i, m in enumerate(messages):
        content = m["content"]
        print(f"  [{i}] role={m['role']}")
        if isinstance(content, list):
            for block in content:
                btype = block.get("type")
                if btype == "image":
                    print("       [image block]")
                elif btype == "text":
                    print(f"       [text] {block.get('text','')[:120]!r}")
        else:
            print(f"       {str(content)[:120]!r}")


def debug_tokenized_sample(sample, idx: int, processor, max_seq_length: int, dbg: dict):
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
        print(f"  fits context : {'✓ YES' if fits else '✗ NO'}")
        print(f"  prompt preview : {text[:200].replace(chr(10), '↵')!r}")
    except Exception as e:
        print(f"  [WARN] tokenization failed: {e}")


def debug_label_distribution(dataset, dbg: dict):
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
        print(f"  GPU {i}: {props.name}")
        print(
            f"    Total={total:.2f}GB  Reserved={reserved:.2f}GB  Allocated={allocated:.2f}GB"
        )


# ── Dataset ───────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def is_video_sample(path: str) -> bool:
    return Path(path).suffix.lower() == ".pt"


def load_json_dataset(json_path) -> list[dict]:
    with open(resolve(json_path), encoding="utf-8") as f:
        return json.load(f)


def make_hf_dataset(records: list[dict]) -> Dataset:
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


def fits_in_context(record: dict, processor, max_len: int) -> bool:
    try:
        msgs = json.loads(record["messages_json"])
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        ids = processor.tokenizer(text, truncation=False)["input_ids"]
        return len(ids) <= max_len
    except Exception:
        return False


# ── Data collator ─────────────────────────────────────────────────────────────


class VLMDataCollator:
    """
    Handles both image samples and video tensor samples (.pt).

    For video .pt files (produced by 00_extract_frames.py):
      - loads the dict with keys "frames" (T,C,H,W) and "grid_thw" (1,3)
      - passes pixel_values_videos + video_grid_thw to the processor
      - the model routes them through its temporal attention path

    For image samples:
      - standard PIL load → processor
    """

    def __init__(self, processor, max_seq_length: int = 2048):
        self.processor = processor
        self.max_seq_length = max_seq_length

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_image(self, path: str) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot load image {path}: {e}")
            return Image.new("RGB", (224, 224), color=(128, 128, 128))

    def _load_video_pt(self, path: str) -> dict:
        """Load a .pt temporal tensor package produced by 00_extract_frames.py."""
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
            # data["frames"]   : (T, C, H, W) float32 [0,1]
            # data["grid_thw"] : (1, 3) long
            return data
        except Exception as e:
            print(f"[WARN] Cannot load video pt {path}: {e}")
            # Return a minimal 1-frame dummy so the batch doesn't break
            dummy_frames = torch.zeros(2, 3, 224, 224, dtype=torch.float32)
            dummy_grid = torch.tensor([[1, 16, 16]], dtype=torch.long)
            return {"frames": dummy_frames, "grid_thw": dummy_grid}

    # ── collate ──────────────────────────────────────────────────────────────

    def __call__(self, batch: list[dict]) -> dict:
        texts: list[str] = []
        images: list[Image.Image | None] = []
        video_tensors: list[torch.Tensor | None] = []  # (T,C,H,W) per sample
        video_grids: list[torch.Tensor | None] = []  # (1,3) per sample
        has_video = False
        has_image = False

        for sample in batch:
            path = sample["image_path"]
            msgs = json.loads(sample["messages_json"])
            text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

            if is_video_sample(path):
                has_video = True
                vdata = self._load_video_pt(path)
                video_tensors.append(vdata["frames"])  # (T,C,H,W)
                video_grids.append(vdata["grid_thw"])  # (1,3)
                images.append(None)
            else:
                has_image = True
                images.append(self._load_image(path))
                video_tensors.append(None)
                video_grids.append(None)

        # ── Build processor inputs ────────────────────────────────────────────
        # We call the processor separately for image vs video sub-batches,
        # then merge, because the processor's image / video kwargs differ.

        # Pure-image batch (most common case)
        if has_image and not has_video:
            encoding = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

        # Pure-video batch
        elif has_video and not has_image:
            # Stack frames: all samples must share T,H,W after padding in 00_extract
            # Concatenate along batch (each video is independent)
            # pixel_values_videos expected shape: (sum_of_tokens, C)
            # We flatten (T*grid_h*grid_w patches) per video
            pv_videos, grid_thw_list = self._build_video_inputs(
                video_tensors, video_grids
            )
            encoding = self.processor(
                text=texts,
                videos=None,  # raw PIL videos not used; we pass tensors directly
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            encoding["pixel_values_videos"] = pv_videos
            encoding["video_grid_thw"] = grid_thw_list

        # Mixed batch (image + video)  — split and merge
        else:
            image_indices = [i for i, t in enumerate(video_tensors) if t is None]
            video_indices = [i for i, t in enumerate(video_tensors) if t is not None]

            img_enc = (
                self.processor(
                    text=[texts[i] for i in image_indices],
                    images=[images[i] for i in image_indices],
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                if image_indices
                else None
            )

            vid_pv, vid_grid = (
                self._build_video_inputs(
                    [video_tensors[i] for i in video_indices],
                    [video_grids[i] for i in video_indices],
                )
                if video_indices
                else (None, None)
            )

            vid_enc = (
                self.processor(
                    text=[texts[i] for i in video_indices],
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                if video_indices
                else None
            )
            if vid_enc is not None and vid_pv is not None:
                vid_enc["pixel_values_videos"] = vid_pv
                vid_enc["video_grid_thw"] = vid_grid

            encoding = self._merge_encodings(
                img_enc, vid_enc, image_indices, video_indices, len(batch)
            )

        # ── Labels ───────────────────────────────────────────────────────────
        labels = encoding["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoding["labels"] = labels
        return encoding

    def _build_video_inputs(
        self,
        video_tensors: list[torch.Tensor],
        video_grids: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert list of (T,C,H,W) float32 tensors to the flat token layout
        Qwen3.5 expects for pixel_values_videos.

        Qwen3.5 VisionPatchEmbed expects:
          input shape: (num_patches, C, temporal_patch_size, patch_size, patch_size)

        We produce a simpler approximation here — flatten T*H_patches*W_patches
        and let the model's internal patch embedding handle the rest.
        The processor's feature_extractor is not used for pre-extracted tensors.
        """
        all_frames_flat: list[torch.Tensor] = []
        all_grids: list[torch.Tensor] = []

        for frames, grid in zip(video_tensors, video_grids):
            # frames: (T, C, H, W) — already float32 [0,1]
            all_frames_flat.append(frames)  # keep per-video for now
            all_grids.append(grid)  # (1, 3)

        # pixel_values_videos: concatenate all video frames along dim=0
        # shape: (total_T_all_videos, C, H, W)
        pv = torch.cat(all_frames_flat, dim=0).to(torch.float32)

        # video_grid_thw: (num_videos, 3)
        grid_thw = torch.cat(all_grids, dim=0)  # (num_videos, 3)

        return pv, grid_thw

    @staticmethod
    def _merge_encodings(
        img_enc,
        vid_enc,
        image_indices: list[int],
        video_indices: list[int],
        total: int,
    ) -> dict:
        """Merge image and video encodings back into a single dict in original order."""
        if img_enc is None:
            return vid_enc
        if vid_enc is None:
            return img_enc

        merged = {}
        # Handle input_ids and attention_mask by index reconstruction
        # For simplicity, we concatenate and sort by original order
        all_keys = set(img_enc.keys()) | set(vid_enc.keys())
        for key in all_keys:
            iv = img_enc.get(key)
            vv = vid_enc.get(key)
            if iv is None:
                merged[key] = vv
            elif vv is None:
                merged[key] = iv
            elif isinstance(iv, torch.Tensor) and isinstance(vv, torch.Tensor):
                merged[key] = torch.cat([iv, vv], dim=0)
            else:
                merged[key] = iv
        return merged


# ── Model ─────────────────────────────────────────────────────────────────────


def load_model_and_processor(cfg: dict):
    model_name = cfg["model"]["name"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # float32 compute dtype for numerical stability
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model via AutoModelForImageTextToText: {model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.float32,  # float32 throughout
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Cast vision encoder to float32 explicitly
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        model.model.visual = model.model.visual.to(torch.float32)
        print("  Cast vision encoder to float32")
    elif hasattr(model, "visual"):
        model.visual = model.visual.to(torch.float32)
        print("  Cast vision encoder to float32")

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


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train with TRL SFTTrainer")
    parser.add_argument("--config", default="config/config_trl.yaml")
    parser.add_argument("--base_config", default="config/config_base.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("extra", nargs="*", help="Extra flags e.g. debug=on")
    args = parser.parse_args()

    extra_flags = {k: v for k, v in (f.split("=", 1) for f in args.extra if "=" in f)}
    cli_debug = extra_flags.get("debug", "").lower()

    base_cfg = load_config(args.base_config)
    trl_cfg = load_config(args.config)
    cfg = merge_configs(base_cfg, trl_cfg)

    train_cfg = cfg["training"]
    ds_cfg = cfg["dataset"]
    smoke_cfg = cfg.get("smoke_test", {})

    dbg = cfg.get("debug", {"enabled": False})
    if cli_debug == "on":
        dbg["enabled"] = True
    elif cli_debug == "off":
        dbg["enabled"] = False

    smoke_mode = dbg.get("enabled", False)

    if smoke_mode:
        print("\n" + "=" * 60)
        print("  SMOKE TEST / DEBUG MODE ON")
        print("=" * 60)
    else:
        print("\n[DEBUG OFF] Full training run.")

    # ── Load data ─────────────────────────────────────────────────────────────
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

    n_dbg = dbg.get("n_samples", 2)
    for i in range(min(n_dbg, len(train_dataset))):
        debug_raw_sample(train_dataset[i], idx=i, dbg=dbg)

    debug_label_distribution(train_records, dbg=dbg)

    # ── Context-length filter (use a lightweight pre-check processor) ─────────
    _proc = AutoProcessor.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    if _proc.tokenizer.pad_token is None:
        _proc.tokenizer.pad_token = _proc.tokenizer.eos_token
    max_len = cfg["model"]["max_seq_length"]

    before = len(train_dataset)
    train_dataset = train_dataset.filter(lambda s: fits_in_context(s, _proc, max_len))
    val_dataset = val_dataset.filter(lambda s: fits_in_context(s, _proc, max_len))
    print(
        f"  After context filter → Train: {len(train_dataset)}  Val: {len(val_dataset)}"
        f"  (removed {before - len(train_dataset)} long samples)"
    )
    del _proc

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model_and_processor(cfg)
    model = apply_lora(model, cfg)

    for i in range(min(n_dbg, len(train_dataset))):
        debug_tokenized_sample(
            train_dataset[i],
            idx=i,
            processor=processor,
            max_seq_length=max_len,
            dbg=dbg,
        )

    debug_gpu(dbg=dbg)

    # ── Collator ──────────────────────────────────────────────────────────────
    collator = VLMDataCollator(processor, max_seq_length=max_len)

    # ── TrainingArguments ─────────────────────────────────────────────────────
    output_dir = str(resolve(train_cfg["output_dir"]))
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
    log_steps = (
        smoke_cfg.get("logging_steps", train_cfg["logging_steps"])
        if smoke_mode
        else train_cfg["logging_steps"]
    )

    if smoke_mode:
        print(
            f"\n  [SMOKE] epochs={num_epochs}  max_steps={max_steps}  save_steps={save_steps}"
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
        fp16=False,  # disabled — we use float32
        save_strategy=train_cfg["save_strategy"],
        save_steps=save_steps,
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=eval_steps,
        logging_steps=log_steps,
        load_best_model_at_end=False,
        report_to=train_cfg["report_to"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        local_rank=-1,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print("\nStarting training ...")
    trainer.train(resume_from_checkpoint=output_dir if args.resume else None)

    # ── Save ──────────────────────────────────────────────────────────────────
    final_path = Path(output_dir) / "final_adapter"
    trainer.model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n✓ Adapter saved: {final_path}")

    if smoke_mode:
        print("\n" + "=" * 60)
        print("  ✓ SMOKE TEST PASSED — load → train → save complete.")
        print("=" * 60)
    else:
        print("  Next: eval/eval_trl.py  or  merge_trl.py")


if __name__ == "__main__":
    main()
