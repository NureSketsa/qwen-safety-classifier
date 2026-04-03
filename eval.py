"""
eval.py
=======
Evaluate a fine-tuned checkpoint against the validation set.

Metrics computed:
  - F1-Score (macro & binary UNSAFE)
  - Recall for UNSAFE class (priority metric)
  - Precision for UNSAFE class
  - AUC-ROC
  - BERTScore for reasoning quality (Indonesian)
  - Confusion matrix

Usage:
  python eval.py --checkpoint output/trl_checkpoint/final_adapter --config config/config_trl.yaml
  python eval.py --checkpoint output/unsloth_merged --merged      --config config/config_unsloth.yaml
  python eval.py --checkpoint output/trl_checkpoint/final_adapter --n 50  # manual eval subset
"""

import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm


# ── Config ──────────────────────────────────────────────────────────────────


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


# ── Inference ────────────────────────────────────────────────────────────────


def load_model(checkpoint: str, merged: bool, cfg: dict):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    model_name = cfg["model"]["name"]

    if merged:
        print(f"Loading merged model from: {checkpoint}")
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    else:
        print(f"Loading base model: {model_name}")
        print(f"Loading adapter:    {checkpoint}")
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, checkpoint)
        model = model.merge_and_unload()
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    model.eval()
    return model, processor


def run_inference(model, processor, sample: dict, max_new_tokens: int = 512) -> str:
    img_path = sample["image_path"]
    messages = sample["messages"]

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Cannot load {img_path}: {e}")
        image = Image.new("RGB", (224, 224), (128, 128, 128))

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
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
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

    reasoning_match = re.search(
        r"REASONING\s*:\s*(.+?)(?=\nLABEL\s*:|$)", text, re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return label, reasoning


# ── Metrics ──────────────────────────────────────────────────────────────────


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
        print("[WARN] bert-score not installed. Skipping BERTScore.")
        return {}
    except Exception as e:
        print(f"[WARN] BERTScore failed: {e}")
        return {}


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned safety classifier"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to adapter or merged model dir"
    )
    parser.add_argument("--config", default="config/config_trl.yaml")
    parser.add_argument("--base_config", default="config/config_base.yaml")
    parser.add_argument(
        "--merged", action="store_true", help="Checkpoint is a merged model"
    )
    parser.add_argument("--n", type=int, default=None, help="Limit to N samples")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    args = parser.parse_args()

    # Load and merge configs
    base_cfg = load_config(args.base_config)
    specific_cfg = load_config(args.config)
    cfg = merge_configs(base_cfg, specific_cfg)

    ds_cfg = cfg["dataset"]
    ev_cfg = cfg["eval"]

    split_path = ds_cfg["val_json"] if args.split == "val" else ds_cfg["train_json"]

    print(f"Loading {args.split} set: {split_path}")
    with open(split_path, encoding="utf-8") as f:
        records = json.load(f)

    if args.n:
        records = records[: args.n]
        print(f"Evaluating first {args.n} samples")

    # ── Load model
    model, processor = load_model(args.checkpoint, args.merged, cfg)
    max_new_tokens = cfg["model"]["max_new_tokens"]

    # ── Run inference
    y_true_labels = []
    y_pred_labels = []
    pred_reasonings = []
    ref_reasonings = []

    for sample in tqdm(records, desc="Inference"):
        gt_label = sample.get("label", "UNKNOWN").upper()
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
    results_dir = Path(
        ev_cfg["results_dir"]
    )  # ← trl → .../trl  |  unsloth → .../unsloth
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_samples": len(records),
        "classification": {k: v for k, v in cls_metrics.items() if k != "report"},
        "bertscore": bs_metrics,
    }

    out_path = (
        results_dir / f"eval_{Path(args.checkpoint).parent.name}_{args.split}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved: {out_path}")


if __name__ == "__main__":
    main()
