"""
00_extract_frames.py
====================
Extract frames from video files inside dataset/Kontent/class_X/
and save them as images alongside other images.

Naming convention kept consistent with existing images:
  X_Y.jpg  →  already images, skip
  video files inside class_X/  →  extract as  X_<videoname>_f<N>.jpg

After extraction, the extracted frames can be added to metadata.csv
by the dataset team with the correct CLASSIFICATION and REASONING.

Usage:
  python 00_extract_frames.py
  python 00_extract_frames.py --fps 1 --ext mp4,avi,mov
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_video_files(class_dir: Path, extensions: set[str]) -> list[Path]:
    return [p for p in class_dir.iterdir() if p.suffix.lower() in extensions]


def extract_frames_ffmpeg(
    video_path: Path,
    output_dir: Path,
    class_idx: int,
    fps: float = 1.0,
) -> list[Path]:
    """
    Extract frames at `fps` rate using ffmpeg.
    Output pattern:  class_idx_<stem>_f%04d.jpg
    Returns list of extracted frame paths.
    """
    stem = re.sub(r"[^\w]", "_", video_path.stem)
    pattern = output_dir / f"{class_idx}_{stem}_f%04d.jpg"

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",  # JPEG quality (2 = high)
        "-y",  # overwrite
        str(pattern),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] ffmpeg error on {video_path.name}:\n{result.stderr[-300:]}")
        return []

    # Collect output frames
    prefix = f"{class_idx}_{stem}_f"
    frames = sorted(output_dir.glob(f"{prefix}*.jpg"))
    return frames


def process_class_dir(
    class_dir: Path,
    class_idx: int,
    fps: float,
    video_exts: set[str],
) -> int:
    videos = get_video_files(class_dir, video_exts)
    if not videos:
        return 0

    total = 0
    for video in videos:
        print(f"  Extracting {video.name} @ {fps}fps ...")
        frames = extract_frames_ffmpeg(video, class_dir, class_idx, fps)
        print(f"    → {len(frames)} frames saved")
        total += len(frames)

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Extract video frames for training dataset"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--fps", type=float, default=1.0, help="Frames per second to extract"
    )
    parser.add_argument(
        "--ext",
        default="mp4,avi,mov,mkv,webm,flv",
        help="Comma-separated video extensions (without dot)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_root = Path(cfg["dataset"]["image_root"])
    class_folders: dict = cfg["dataset"]["class_folders"]
    video_exts = {"." + e.strip().lower() for e in args.ext.split(",")}

    if not image_root.exists():
        print(f"[ERROR] image_root not found: {image_root}")
        sys.exit(1)

    # Check ffmpeg available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            "[ERROR] ffmpeg not found. Install with: apt install ffmpeg  or  brew install ffmpeg"
        )
        sys.exit(1)

    total_extracted = 0
    for idx_str, folder_name in class_folders.items():
        class_dir = image_root / folder_name
        if not class_dir.exists():
            print(f"[SKIP] {class_dir} does not exist")
            continue

        print(f"\n[class_{idx_str}] → {class_dir}")
        count = process_class_dir(class_dir, int(idx_str), args.fps, video_exts)
        total_extracted += count

    print(f"\n✓ Done. Total frames extracted: {total_extracted}")
    print(
        "  Next: add new frames to dataset/metadata.csv with CLASSIFICATION + REASONING"
    )


if __name__ == "__main__":
    main()
