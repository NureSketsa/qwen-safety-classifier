"""
01_prepare_dataset.py
=====================
Convert metadata.csv + image folders into train_data.json / val_data.json
in the chat-template format expected by TRL / Unsloth SFTTrainer.

CSV format expected:
  image_name,CLASSIFICATION,kategori,REASONING
  1_126.jpg,UNSAFE,HENTAI,"Berdasarkan ..."

Image naming convention:
  X_Y.jpg  →  folder class_X, file Y.jpg
  (X is the integer prefix before the first underscore)

Output JSON format (one entry per line is also fine — we write a JSON array):
  [
    {
      "messages": [
        {"role": "system",    "content": "<system prompt>"},
        {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
        {"role": "assistant", "content": "REASONING: ...\nLABEL: SAFE"}
      ],
      "image_path": "dataset/Kontent/class_1/126.jpg"
    },
    ...
  ]

Usage:
  python 01_prepare_dataset.py
  python 01_prepare_dataset.py --config config.yaml --verify
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# ── Helpers ─────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_image_name(
    image_name: str, class_folders: dict, image_root: Path
) -> Path | None:
    """
    Parse '1_126.jpg' → image_root/class_1/1_126.jpg
    (KEEP full filename, only use prefix to find folder)
    """
    name = Path(image_name).name

    parts = name.split("_", 1)
    if len(parts) != 2:
        return None

    class_idx_str = parts[0]

    try:
        class_idx = int(class_idx_str)
    except ValueError:
        return None

    folder_name = class_folders.get(class_idx) or class_folders.get(str(class_idx))
    if folder_name is None:
        return None

    # ✅ KEEP FULL NAME
    return image_root / folder_name / name


def build_messages(row: pd.Series, system_prompt: str) -> dict:
    """
    Build the chat messages list for one sample.
    The user turn contains an image placeholder + instruction text.
    The assistant turn is the expected model output.
    """
    label = str(row["CLASSIFICATION"]).strip().upper()
    reasoning = str(row["REASONING"]).strip()

    user_content = [
        {"type": "image"},
        {
            "type": "text",
            "text": (
                "Analisis gambar berikut dan tentukan apakah konten tersebut "
                "melanggar UU ITE Pasal 28 Ayat 2. "
                "Berikan reasoning detail dalam Bahasa Indonesia dan label SAFE atau UNSAFE."
            ),
        },
    ]

    assistant_content = f"REASONING: {reasoning}\nLABEL: {label}"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for QLoRA fine-tuning"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check that all image files actually exist on disk",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    system_prompt: str = cfg["prompt"]["system"].strip()

    metadata_csv = Path(ds_cfg["metadata_csv"])
    image_root = Path(ds_cfg["image_root"])
    train_json = Path(ds_cfg["train_json"])
    val_json = Path(ds_cfg["val_json"])
    val_split = float(ds_cfg["val_split"])
    seed = int(ds_cfg["seed"])
    class_folders: dict = {int(k): v for k, v in ds_cfg["class_folders"].items()}

    # ── Load CSV ────────────────────────────────────────────────────────────
    if not metadata_csv.exists():
        print(f"[ERROR] metadata.csv not found: {metadata_csv}")
        sys.exit(1)

    df = pd.read_csv(metadata_csv)
    print(f"Loaded {len(df)} rows from {metadata_csv}")

    required_cols = {"image_name", "CLASSIFICATION", "REASONING"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns in CSV: {missing}")
        print(f"        Found columns: {list(df.columns)}")
        sys.exit(1)

    # Drop rows with missing critical fields
    before = len(df)
    df = df.dropna(subset=["image_name", "CLASSIFICATION", "REASONING"])
    if len(df) < before:
        print(f"[WARN] Dropped {before - len(df)} rows with missing values")

    # ── Resolve image paths ──────────────────────────────────────────────────
    records = []
    skipped = 0

    for _, row in df.iterrows():
        img_path = parse_image_name(row["image_name"], class_folders, image_root)

        if img_path is None:
            print(f"[WARN] Cannot parse image name: {row['image_name']!r}")
            skipped += 1
            continue

        if args.verify and not img_path.exists():
            print(f"[WARN] Image not found on disk: {img_path}")
            skipped += 1
            continue

        entry = build_messages(row, system_prompt)
        entry["image_path"] = str(img_path)
        entry["label"] = str(row["CLASSIFICATION"]).strip().upper()
        records.append(entry)

    print(f"\nValid records : {len(records)}")
    print(f"Skipped       : {skipped}")

    if len(records) == 0:
        print("[ERROR] No valid records. Check CSV and image paths.")
        sys.exit(1)

    # ── Label distribution ───────────────────────────────────────────────────
    labels = [r["label"] for r in records]
    label_counts = pd.Series(labels).value_counts()
    print("\nLabel distribution:")
    for lbl, cnt in label_counts.items():
        print(f"  {lbl:<10} {cnt:>6}  ({cnt/len(records)*100:.1f}%)")

    # ── Train / Val split ────────────────────────────────────────────────────
    train_records, val_records = train_test_split(
        records,
        test_size=val_split,
        random_state=seed,
        stratify=labels,  # keep label ratio in both splits
    )
    print(f"\nTrain: {len(train_records)}  |  Val: {len(val_records)}")

    # ── Write JSON ───────────────────────────────────────────────────────────
    train_json.parent.mkdir(parents=True, exist_ok=True)

    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)

    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved: {train_json}  ({len(train_records)} samples)")
    print(f"✓ Saved: {val_json}   ({len(val_records)} samples)")
    print("\nNext: run train_trl.py or train_unsloth.py")


if __name__ == "__main__":
    main()
