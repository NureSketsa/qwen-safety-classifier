"""
merge_trl.py
============
Merge a LoRA adapter into the base model and save as a standalone model.

Changes vs. original:
  - Uses AutoModelForImageTextToText (not Qwen3_5ForConditionalGeneration)
  - Uses torch.float32 for merge (float32 is required for accurate weight addition)

Usage:
  python merge_trl.py
  python merge_trl.py --adapter output/trl_checkpoint/final_adapter --output output/trl_merged
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="output/trl_checkpoint/final_adapter")
    parser.add_argument("--output", default="output/trl_merged")
    parser.add_argument("--config", default="config/config_trl.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model (float32): {model_name}")
    # float32 is required for merge — quantized (int4) weights cannot be merged
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging adapter into base model ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)

    processor = AutoProcessor.from_pretrained(args.adapter, trust_remote_code=True)
    processor.save_pretrained(output_dir)

    print("✓ Merge complete.")


if __name__ == "__main__":
    main()
