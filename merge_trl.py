import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import yaml
import argparse
from pathlib import Path
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    BitsAndBytesConfig,
)
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

    print(f"Loading base model: {model_name}")
    # Load in fp16, NO quantization — merging requires real weights
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
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
