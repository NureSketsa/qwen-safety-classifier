"""
eval_trl.py
===========
Evaluate a TRL fine-tuned checkpoint against the validation set.

Changes vs. original:
  - Uses AutoModelForImageTextToText (not Qwen3_5ForConditionalGeneration)
  - Uses torch.float32 for compute dtype
  - Handles both image samples and video .pt tensor samples

Usage:
  python eval/eval_trl.py --checkpoint output/trl_checkpoint/final_adapter
  python eval/eval_trl.py --checkpoint output/trl_merged --merged
  python eval/eval_trl.py --checkpoint output/trl_checkpoint/final_adapter --n 50
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


# ── Config ───────────────────────────────────────────────────────────────────


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


# ── Model loader ──────────────────────────────────────────────────────────────


def load_model(checkpoint: str, merged: bool, cfg: dict):
    model_name = cfg["model"]["name"]
    checkpoint_path = str(resolve(checkpoint))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,  # float32
        bnb_4bit_use_double_quant=True,
    )

    if merged:
        print(f"Loading merged model: {checkpoint_path}")
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint_path,
            quantization_config=bnb_config,
            device_map="cuda:0",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
    else:
        print(f"Loading base: {model_name}  adapter: {checkpoint_path}")
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda:0",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, checkpoint_path)
        processor = AutoProcessor.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model.eval()
    print("  Model ready.")
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────


def _load_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Cannot load {path}: {e}")
        return Image.new("RGB", (224, 224), (128, 128, 128))


def _load_video_pt(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"[WARN] Cannot load video pt {path}: {e}")
        return {
            "frames": torch.zeros(2, 3, 224, 224, dtype=torch.float32),
            "grid_thw": torch.tensor([[1, 16, 16]], dtype=torch.long),
        }


def run_inference(model, processor, sample: dict, max_new_tokens: int = 256) -> str:
    img_path = sample["image_path"]
    messages = [m for m in sample["messages"] if m["role"] != "assistant"]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    is_video = Path(img_path).suffix.lower() == ".pt"

    if is_video:
        vdata = _load_video_pt(img_path)
        frames = vdata["frames"].to(torch.float32)  # (T,C,H,W)
        grid_thw = vdata["grid_thw"]  # (1,3)

        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to("cuda:0")
        # Inject video tensors directly
        inputs["pixel_values_videos"] = frames.to("cuda:0")
        inputs["video_grid_thw"] = grid_thw.to("cuda:0")
    else:
        image = _load_image(img_path)
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to("cuda:0")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def parse_output(text: str) -> tuple[str, str]:
    label = "UNKNOWN"
    reasoning = text
    label_match = re.search(r"LABEL\s*:\s*(SAFE|UNSAFE)", text, re.IGNORECASE)
    if label_match:
        label = label_match.group(1).upper()
    r_match = re.search(
        r"REASONING\s*:\s*(.+?)(?=\nLABEL\s*:|$)", text, re.DOTALL | re.IGNORECASE
    )
    if r_match:
        reasoning = r_match.group(1).strip()
    return label, reasoning


# ── Metrics ───────────────────────────────────────────────────────────────────


def compute_classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        recall_score,
        precision_score,
        roc_auc_score,
    )

    lmap = {"UNSAFE": 1, "SAFE": 0, "UNKNOWN": 0}
    yt = [lmap.get(l, 0) for l in y_true]
    yp = [lmap.get(l, 0) for l in y_pred]
    results = {
        "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
        "f1_unsafe": f1_score(yt, yp, pos_label=1, zero_division=0),
        "recall_unsafe": recall_score(yt, yp, pos_label=1, zero_division=0),
        "precision_unsafe": precision_score(yt, yp, pos_label=1, zero_division=0),
        "auc_roc": roc_auc_score(yt, yp) if len(set(yt)) > 1 else None,
        "confusion_matrix": confusion_matrix(yt, yp, labels=[0, 1]).tolist(),
        "report": classification_report(
            yt, yp, target_names=["SAFE", "UNSAFE"], zero_division=0
        ),
    }
    return results


def compute_bertscore(preds: list[str], refs: list[str], lang: str = "id") -> dict:
    try:
        from bert_score import score as bscore

        P, R, F = bscore(preds, refs, lang=lang, verbose=False)
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F.mean().item(),
        }
    except ImportError:
        print("[WARN] bert-score not installed.")
        return {}
    except Exception as e:
        print(f"[WARN] BERTScore failed: {e}")
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate TRL safety classifier")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/config_trl.yaml")
    parser.add_argument("--base_config", default="config/config_base.yaml")
    parser.add_argument("--merged", action="store_true")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    cfg = merge_configs(base_cfg, load_config(args.config))

    ds_cfg = cfg["dataset"]
    ev_cfg = cfg["eval"]

    split_path = resolve(
        ds_cfg["val_json"] if args.split == "val" else ds_cfg["train_json"]
    )
    print(f"Loading {args.split} set: {split_path}")
    with open(split_path, encoding="utf-8") as f:
        records = json.load(f)

    if args.n:
        records = records[: args.n]
        print(f"Limiting to {args.n} samples")

    model, processor = load_model(args.checkpoint, args.merged, cfg)
    max_new_tokens = cfg["model"]["max_new_tokens"]

    y_true, y_pred = [], []
    pred_r, ref_r = [], []

    for sample in tqdm(records, desc="Inference"):
        gt_label = sample.get("label", "UNKNOWN").upper()
        gt_reasoning = ""
        for msg in sample["messages"]:
            if msg["role"] == "assistant":
                m = re.search(
                    r"REASONING\s*:\s*(.+?)(?=\nLABEL\s*:|$)",
                    msg["content"],
                    re.DOTALL | re.IGNORECASE,
                )
                if m:
                    gt_reasoning = m.group(1).strip()

        raw = run_inference(model, processor, sample, max_new_tokens)
        pred_label, pred_reasoning = parse_output(raw)

        y_true.append(gt_label)
        y_pred.append(pred_label)
        pred_r.append(pred_reasoning)
        ref_r.append(gt_reasoning)

    print("\n" + "=" * 60 + "\nCLASSIFICATION METRICS\n" + "=" * 60)
    cls = compute_classification_metrics(y_true, y_pred)
    print(cls["report"])
    print(f"F1 Macro         : {cls['f1_macro']:.4f}")
    print(f"F1 UNSAFE        : {cls['f1_unsafe']:.4f}")
    print(f"Recall UNSAFE    : {cls['recall_unsafe']:.4f}  ← priority")
    print(f"Precision UNSAFE : {cls['precision_unsafe']:.4f}")
    if cls["auc_roc"] is not None:
        print(f"AUC-ROC          : {cls['auc_roc']:.4f}")
    print(f"Confusion matrix : {cls['confusion_matrix']}")

    n_unknown = y_pred.count("UNKNOWN")
    if n_unknown:
        print(f"[WARN] {n_unknown}/{len(y_pred)} unparseable predictions.")

    print("\n" + "=" * 60 + "\nBERTSCORE\n" + "=" * 60)
    bs = compute_bertscore(pred_r, ref_r, ev_cfg["bertscore_lang"])
    if bs:
        print(f"Precision : {bs['bertscore_precision']:.4f}")
        print(f"Recall    : {bs['bertscore_recall']:.4f}")
        print(f"F1        : {bs['bertscore_f1']:.4f}")

    results_dir = resolve(ev_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"eval_{Path(args.checkpoint).name}_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "merged": args.merged,
                "split": args.split,
                "n_samples": len(records),
                "n_unknown_preds": n_unknown,
                "classification": {k: v for k, v in cls.items() if k != "report"},
                "bertscore": bs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n✓ Results saved: {out_path}")


if __name__ == "__main__":
    main()
