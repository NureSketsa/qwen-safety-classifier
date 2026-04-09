"""
00_extract_frames.py
====================
Extract video files inside dataset/Kontent/class_X/ and save them as
**temporal video tensors** — NOT flat frame images.

Why temporal tensors instead of flat frames?
  Qwen3.5's vision encoder has a full temporal attention path:
    - pixel_values_videos  : (total_frames * H * W patches, C)
    - video_grid_thw       : (num_videos, 3)  → (T, H, W) grid per video

  Flat frame images lose the temporal relationship between frames.
  By saving the entire video as a single (T, C, H, W) tensor package,
  the model can attend across frames and extract motion/context cues
  that are invisible when frames are treated as independent images.

Output per video:
  class_X/<videoname>_video.pt   ← dict with keys:
      "frames"      : torch.float32 tensor (T, C, H, W), values in [0, 1]
      "grid_thw"    : (T, H_patches, W_patches)   — the llm grid shape
      "fps_extracted": float
      "source"      : str  original video filename
      "class_idx"   : int

  metadata_videos.csv is appended/created at dataset root with columns:
      video_pt_path, class_idx, num_frames, grid_t, grid_h, grid_w

Usage:
  python 00_extract_frames.py
  python 00_extract_frames.py --fps 2 --patch_size 14 --temporal_patch_size 2
  python 00_extract_frames.py --ext mp4,avi,mov --max_frames 32
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import yaml

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# Qwen3.5 vision defaults (must match model config)
DEFAULT_PATCH_SIZE = 14  # spatial patch size in pixels
DEFAULT_TEMPORAL_PATCH_SIZE = 2  # frames merged per temporal token
DEFAULT_TARGET_H = 224  # resize height before patching
DEFAULT_TARGET_W = 224  # resize width  before patching
DEFAULT_FPS = 1.0
DEFAULT_MAX_FRAMES = 16  # cap to avoid OOM on long videos


# ── Config ───────────────────────────────────────────────────────────────────


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        # Try config/ subdirectory (repo layout)
        path = Path("config") / "config_base.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# ── Video reading ─────────────────────────────────────────────────────────────


def read_video_cv2(
    video_path: Path,
    fps_target: float,
    max_frames: int,
    target_h: int,
    target_w: int,
) -> tuple[torch.Tensor | None, float]:
    """
    Read video with OpenCV, sample at fps_target, resize, return float32 tensor.

    Returns:
        frames: (T, C, H, W) float32 in [0, 1]  or None on failure
        actual_fps: fps used for sampling
    """
    try:
        import cv2
    except ImportError:
        print("[ERROR] opencv-python not installed. pip install opencv-python-headless")
        return None, 0.0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] cv2 cannot open: {video_path.name}")
        return None, 0.0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute which frame indices to sample
    step = max(1, round(video_fps / fps_target))
    frame_indices = list(range(0, total_frames, step))[:max_frames]

    if len(frame_indices) == 0:
        cap.release()
        print(f"  [WARN] No frames to sample from {video_path.name}")
        return None, 0.0

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # BGR → RGB, resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None, 0.0

    import numpy as np

    arr = np.stack(frames, axis=0)  # (T, H, W, C)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
    return tensor, fps_target


# ── Grid shape helpers ────────────────────────────────────────────────────────


def compute_grid_thw(
    num_frames: int,
    height: int,
    width: int,
    patch_size: int,
    temporal_patch_size: int,
) -> tuple[int, int, int]:
    """
    Compute the (T, H_patches, W_patches) grid that Qwen3.5 will see.

    Qwen3.5 vision uses:
      - spatial: H // patch_size  ×  W // patch_size patches per frame
      - temporal: frames are grouped in temporal_patch_size chunks

    Frames are padded to a multiple of temporal_patch_size.
    """
    # Pad T to multiple of temporal_patch_size
    t_padded = (
        (num_frames + temporal_patch_size - 1) // temporal_patch_size
    ) * temporal_patch_size
    grid_t = t_padded // temporal_patch_size
    grid_h = height // patch_size
    grid_w = width // patch_size
    return grid_t, grid_h, grid_w


def pad_frames_to_temporal(
    frames: torch.Tensor,
    temporal_patch_size: int,
) -> torch.Tensor:
    """
    Pad along the T dimension so T is a multiple of temporal_patch_size.
    Padding duplicates the last frame.
    """
    T = frames.shape[0]
    remainder = T % temporal_patch_size
    if remainder == 0:
        return frames
    pad_len = temporal_patch_size - remainder
    last_frame = frames[-1:].expand(pad_len, -1, -1, -1)
    return torch.cat([frames, last_frame], dim=0)


# ── Core processing ───────────────────────────────────────────────────────────


def process_video(
    video_path: Path,
    class_idx: int,
    fps: float,
    max_frames: int,
    patch_size: int,
    temporal_patch_size: int,
    target_h: int,
    target_w: int,
) -> dict | None:
    """
    Process one video file into a temporal tensor package.

    Returns dict ready to torch.save, or None on failure.
    """
    frames, actual_fps = read_video_cv2(video_path, fps, max_frames, target_h, target_w)
    if frames is None or frames.shape[0] == 0:
        return None

    # Pad to temporal_patch_size multiple
    frames = pad_frames_to_temporal(frames, temporal_patch_size)
    T, C, H, W = frames.shape

    grid_t, grid_h, grid_w = compute_grid_thw(T, H, W, patch_size, temporal_patch_size)

    # grid_thw as a 1-row tensor matching Qwen3.5's expected shape (num_videos, 3)
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)

    print(
        f"    frames={T}  grid=({grid_t},{grid_h},{grid_w})  shape={tuple(frames.shape)}"
    )

    return {
        "frames": frames,  # (T, C, H, W) float32 [0,1]
        "grid_thw": grid_thw,  # (1, 3) long
        "fps_extracted": actual_fps,
        "source": video_path.name,
        "class_idx": class_idx,
    }


def process_class_dir(
    class_dir: Path,
    class_idx: int,
    fps: float,
    max_frames: int,
    patch_size: int,
    temporal_patch_size: int,
    target_h: int,
    target_w: int,
    video_exts: set[str],
) -> list[dict]:
    """Process all videos in a class directory. Returns metadata rows."""
    videos = [p for p in class_dir.iterdir() if p.suffix.lower() in video_exts]
    if not videos:
        return []

    metadata_rows = []
    for video in videos:
        print(f"  Processing {video.name} ...")
        result = process_video(
            video,
            class_idx,
            fps,
            max_frames,
            patch_size,
            temporal_patch_size,
            target_h,
            target_w,
        )
        if result is None:
            print(f"    [SKIP] failed to process {video.name}")
            continue

        # Save as .pt next to the video
        out_stem = video.stem + "_video"
        out_path = class_dir / f"{out_stem}.pt"
        torch.save(result, out_path)
        print(f"    → saved {out_path.name}")

        grid_thw = result["grid_thw"][0]
        metadata_rows.append(
            {
                "video_pt_path": str(out_path),
                "class_idx": class_idx,
                "num_frames": result["frames"].shape[0],
                "grid_t": grid_thw[0].item(),
                "grid_h": grid_thw[1].item(),
                "grid_w": grid_thw[2].item(),
                "source": result["source"],
            }
        )

    return metadata_rows


# ── metadata CSV writer ───────────────────────────────────────────────────────


def write_metadata_csv(rows: list[dict], output_csv: Path):
    fieldnames = [
        "video_pt_path",
        "class_idx",
        "num_frames",
        "grid_t",
        "grid_h",
        "grid_w",
        "source",
    ]
    exists = output_csv.exists()
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Metadata appended → {output_csv}  ({len(rows)} new rows)")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extract videos as temporal tensors for Qwen3.5 temporal attention"
    )
    parser.add_argument("--config", default="config/config_base.yaml")
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Frames per second to sample from each video",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Max frames per video (caps memory usage)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help="Spatial patch size in pixels (must match model, default 14)",
    )
    parser.add_argument(
        "--temporal_patch_size",
        type=int,
        default=DEFAULT_TEMPORAL_PATCH_SIZE,
        help="Frames per temporal token (must match model, default 2)",
    )
    parser.add_argument(
        "--target_h",
        type=int,
        default=DEFAULT_TARGET_H,
        help="Resize height before patching (must be divisible by patch_size)",
    )
    parser.add_argument(
        "--target_w",
        type=int,
        default=DEFAULT_TARGET_W,
        help="Resize width before patching (must be divisible by patch_size)",
    )
    parser.add_argument(
        "--ext",
        default="mp4,avi,mov,mkv,webm,flv",
        help="Comma-separated video extensions (without dot)",
    )
    args = parser.parse_args()

    # Validate dimensions
    assert (
        args.target_h % args.patch_size == 0
    ), f"target_h ({args.target_h}) must be divisible by patch_size ({args.patch_size})"
    assert (
        args.target_w % args.patch_size == 0
    ), f"target_w ({args.target_w}) must be divisible by patch_size ({args.patch_size})"

    cfg = load_config(args.config)
    image_root = Path(cfg["dataset"]["image_root"])
    class_folders: dict = cfg["dataset"]["class_folders"]
    video_exts = {"." + e.strip().lower() for e in args.ext.split(",")}

    if not image_root.exists():
        print(f"[ERROR] image_root not found: {image_root}")
        sys.exit(1)

    metadata_csv = image_root.parent / "metadata_videos.csv"

    print(f"Video root   : {image_root}")
    print(f"FPS          : {args.fps}")
    print(f"Max frames   : {args.max_frames}")
    print(f"Patch size   : {args.patch_size}")
    print(f"Temporal patch: {args.temporal_patch_size}")
    print(f"Resize to    : {args.target_h}×{args.target_w}")
    print()

    all_metadata: list[dict] = []

    for idx_str, folder_name in class_folders.items():
        class_dir = image_root / folder_name
        if not class_dir.exists():
            print(f"[SKIP] {class_dir} does not exist")
            continue

        print(f"\n[class_{idx_str}] → {class_dir}")
        rows = process_class_dir(
            class_dir=class_dir,
            class_idx=int(idx_str),
            fps=args.fps,
            max_frames=args.max_frames,
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
            target_h=args.target_h,
            target_w=args.target_w,
            video_exts=video_exts,
        )
        all_metadata.extend(rows)

    if all_metadata:
        write_metadata_csv(all_metadata, metadata_csv)
    else:
        print("\n[INFO] No videos found / processed.")

    print(f"\nTotal videos processed : {len(all_metadata)}")
    print(
        "\nNext steps:\n"
        "  1. Add new rows from metadata_videos.csv to dataset/metadata.csv\n"
        "     with CLASSIFICATION and REASONING columns.\n"
        "  2. Update 01_prepare_dataset.py to handle .pt video entries\n"
        "     alongside regular image entries."
    )


if __name__ == "__main__":
    main()
