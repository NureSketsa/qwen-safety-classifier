"""
eval_trl.py
===========
Evaluate a TRL fine-tuned checkpoint against the validation set.
No Unsloth dependency — pure HuggingFace transformers + PEFT.

Metrics:
  - F1-Score (macro & binary UNSAFE)
  - Recall for UNSAFE class  ← priority metric
  - Precision for UNSAFE class
  - AUC-ROC
  - BERTScore for reasoning quality (Indonesian)
  - Confusion matrix

Usage:
  # adapter (non-merged)
  python eval_trl.py --checkpoint output/trl_checkpoint/final_adapter --config config/config_trl.yaml

  # merged model
  python eval_trl.py --checkpoint output/trl_merged --merged --config config/config_trl.yaml

  # limit samples
  python eval_trl.py --checkpoint output/trl_checkpoint/final_adapter --n 50
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3_5ForConditionalGeneration,
)


# ── Config ───────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result


# ── Model loader ─────────────────────────────────────────────────────────────


def load_model(checkpoint: str, merged: bool, cfg: dict):
    """
    Load model for inference using plain HuggingFace transformers + PEFT.
    - merged=False : loads base model + LoRA adapter (no merge)
    - merged=True  : loads already-merged model directory
    """
    model_name = cfg["model"]["name"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    if merged:
        print(f"Loading merged model from : {checkpoint}")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            checkpoint,
            quantization_config=bnb_config,
            device_map="cuda:0",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    else:
        print(f"Loading base model  : {model_name}")
        print(f"Loading adapter     : {checkpoint}")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda:0",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, checkpoint)
        # Keep adapter attached (no merge) — saves memory, same results
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model.eval()
    print("  Model ready for inference.")
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────


def run_inference(model, processor, sample: dict, max_new_tokens: int = 256) -> str:
    img_path = sample["image_path"]
    messages = sample["messages"]

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Cannot load {img_path}: {e}")
        image = Image.new("RGB", (224, 224), (128, 128, 128))

    # Strip assistant turn — only prompt
    prompt_messages = [m for m in messages if m["role"] != "assistant"]

    text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg_global["model"]["max_seq_length"],
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
    """Extract LABEL and REASONING from model output."""
    label = "UNKNOWN"
    reasoning = text

    label_match = re.search(r"LABEL\s*:\s*(SAFE|UNSAFE)", text, re.IGNORECASE)
    if label_match:
        label = label_match.group(1).upper()

    reasoning_match = re.search(
        r"REASONING\s*:\s*(.+?)(?=\nLABEL\s*:|$)", text, re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

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

    label_map = {"UNSAFE": 1, "SAFE": 0, "UNKNOWN": 0}
    y_true_bin = [label_map.get(l, 0) for l in y_true]
    y_pred_bin = [label_map.get(l, 0) for l in y_pred]

    results = {
        "f1_macro": f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0),
        "f1_unsafe": f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0),
        "recall_unsafe": recall_score(
            y_true_bin, y_pred_bin, pos_label=1, zero_division=0
        ),
        "precision_unsafe": precision_score(
            y_true_bin, y_pred_bin, pos_label=1, zero_division=0
        ),
    }

    if len(set(y_true_bin)) > 1:
        results["auc_roc"] = roc_auc_score(y_true_bin, y_pred_bin)
    else:
        results["auc_roc"] = None

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    results["confusion_matrix"] = cm.tolist()
    results["report"] = classification_report(
        y_true_bin,
        y_pred_bin,
        target_names=["SAFE", "UNSAFE"],
        zero_division=0,
    )

    return results


def compute_bertscore(
    predictions: list[str], references: list[str], lang: str = "id"
) -> dict:
    try:
        from bert_score import score as bert_score_fn

        P, R, F = bert_score_fn(predictions, references, lang=lang, verbose=False)
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F.mean().item(),
        }
    except ImportError:
        print("[WARN] bert-score not installed. pip install bert-score")
        return {}
    except Exception as e:
        print(f"[WARN] BERTScore failed: {e}")
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────

# Global cfg reference needed inside run_inference for max_seq_length
cfg_global: dict = {}


def main():
    global cfg_global

    parser = argparse.ArgumentParser(
        description="Evaluate TRL fine-tuned safety classifier"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to final_adapter dir or merged model dir",
    )
    parser.add_argument("--config", default="config/config_trl.yaml")
    parser.add_argument(
        "--base_config",
        default="config/config_base.yaml",
        help="Optional base config to merge from",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Pass this flag if checkpoint is a merged (non-adapter) model",
    )
    parser.add_argument("--n", type=int, default=None, help="Limit to N samples")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    args = parser.parse_args()

    # ── Load config
    cfg = load_config(args.config)
    if Path(args.base_config).exists():
        base_cfg = load_config(args.base_config)
        cfg = merge_configs(base_cfg, cfg)
    cfg_global = cfg

    ds_cfg = cfg["dataset"]
    ev_cfg = cfg["eval"]

    # ── Load dataset split
    split_path = ds_cfg["val_json"] if args.split == "val" else ds_cfg["train_json"]
    print(f"Loading {args.split} set: {split_path}")
    with open(split_path, encoding="utf-8") as f:
        records = json.load(f)

    if args.n:
        records = records[: args.n]
        print(f"Limiting to first {args.n} samples")

    print(f"Total samples to evaluate: {len(records)}")

    # ── Load model
    model, processor = load_model(args.checkpoint, args.merged, cfg)
    max_new_tokens = cfg["model"]["max_new_tokens"]

    # ── Run inference
    y_true_labels: list[str] = []
    y_pred_labels: list[str] = []
    pred_reasonings: list[str] = []
    ref_reasonings: list[str] = []

    for sample in tqdm(records, desc="Inference"):
        gt_label = sample.get("label", "UNKNOWN").upper()

        # Extract ground-truth reasoning from assistant turn
        gt_reasoning = ""
        for msg in sample["messages"]:
            if msg["role"] == "assistant":
                content = msg["content"]
                r_match = re.search(
                    r"REASONING\s*:\s*(.+?)(?=\nLABEL\s*:|$)",
                    content,
                    re.DOTALL | re.IGNORECASE,
                )
                if r_match:
                    gt_reasoning = r_match.group(1).strip()

        raw_output = run_inference(model, processor, sample, max_new_tokens)
        pred_label, pred_reasoning = parse_output(raw_output)

        y_true_labels.append(gt_label)
        y_pred_labels.append(pred_label)
        pred_reasonings.append(pred_reasoning)
        ref_reasonings.append(gt_reasoning)

    # ── Classification metrics
    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)
    cls_metrics = compute_classification_metrics(y_true_labels, y_pred_labels)
    print(cls_metrics["report"])
    print(f"F1 Macro         : {cls_metrics['f1_macro']:.4f}")
    print(f"F1 UNSAFE        : {cls_metrics['f1_unsafe']:.4f}")
    print(f"Recall UNSAFE    : {cls_metrics['recall_unsafe']:.4f}  ← priority metric")
    print(f"Precision UNSAFE : {cls_metrics['precision_unsafe']:.4f}")
    if cls_metrics["auc_roc"] is not None:
        print(f"AUC-ROC          : {cls_metrics['auc_roc']:.4f}")
    print(
        f"Confusion matrix :\n  [[TN, FP]\n   [FN, TP]] = {cls_metrics['confusion_matrix']}"
    )

    # ── Unknown label report
    n_unknown = y_pred_labels.count("UNKNOWN")
    if n_unknown:
        print(
            f"\n[WARN] {n_unknown}/{len(y_pred_labels)} predictions had no parseable LABEL."
        )
        print(
            "       Check model output format — expected: LABEL: SAFE or LABEL: UNSAFE"
        )

    # ── BERTScore
    print("\n" + "=" * 60)
    print("BERTSCORE (reasoning quality, Indonesian)")
    print("=" * 60)
    bs_metrics = compute_bertscore(
        pred_reasonings, ref_reasonings, lang=ev_cfg["bertscore_lang"]
    )
    if bs_metrics:
        print(f"BERTScore Precision : {bs_metrics['bertscore_precision']:.4f}")
        print(f"BERTScore Recall    : {bs_metrics['bertscore_recall']:.4f}")
        print(f"BERTScore F1        : {bs_metrics['bertscore_f1']:.4f}")

    # ── Save results
    results_dir = Path(ev_cfg["results_dir"]) / "trl"
    results_dir.mkdir(parents=True, exist_ok=True)

    ckpt_tag = Path(args.checkpoint).name  # e.g. "final_adapter" or "trl_merged"
    out_path = results_dir / f"eval_{ckpt_tag}_{args.split}.json"

    results = {
        "checkpoint": args.checkpoint,
        "merged": args.merged,
        "split": args.split,
        "n_samples": len(records),
        "n_unknown_preds": n_unknown,
        "classification": {k: v for k, v in cls_metrics.items() if k != "report"},
        "bertscore": bs_metrics,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved: {out_path}")


if __name__ == "__main__":
    main()
