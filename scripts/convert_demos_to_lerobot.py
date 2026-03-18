"""
HDF5 デモ → LeRobot v2 データセット変換スクリプト

openpi は lerobot.LeRobotDataset 形式を期待するため、v2 フォーマットで保存する。
  meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl, meta/stats.json
  data/chunk-000/episode_XXXXXX.parquet
  images/{front,wrist}/episode_XXXXXX/frame_{:06d}.png

Usage (inside container):
  python scripts/convert_demos_to_lerobot.py \
    --demos /opt/pickcube_demos/demos.h5 \
    --out   /opt/pickcube_lerobot \
    --repo-id pickcube_lerobot
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image as PILImage

parser = argparse.ArgumentParser()
parser.add_argument("--demos",   type=str, default="/opt/pickcube_demos/demos.h5")
parser.add_argument("--out",     type=str, default="/opt/pickcube_lerobot")
parser.add_argument("--repo-id", type=str, default="pickcube_lerobot",
                    help="Logical repo_id (no slashes). Used as directory name under HF_LEROBOT_HOME.")
parser.add_argument("--fps",     type=int, default=10)
parser.add_argument("--task-desc", type=str, default="pick up the red cube")
parser.add_argument("--cam-size",  type=int, default=224)
args = parser.parse_args()

OUT_DIR = Path(args.out)

# Remove existing directory to ensure clean creation
if OUT_DIR.exists():
    print(f"[convert] Removing existing {OUT_DIR} ...")
    shutil.rmtree(OUT_DIR)

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

CAM = args.cam_size
FEATURES = {
    "observation.images.front": {
        "dtype": "image",
        "shape": (CAM, CAM, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist": {
        "dtype": "image",
        "shape": (CAM, CAM, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["eef_x", "eef_y", "eef_z", "eef_qx", "eef_qy", "eef_qz", "eef_qw", "gripper"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
    },
}

print(f"[convert] Creating LeRobot v2 dataset at {OUT_DIR} (repo_id={args.repo_id}) ...")

ds = LeRobotDataset.create(
    repo_id=args.repo_id,
    fps=args.fps,
    root=OUT_DIR,
    features=FEATURES,
    use_videos=False,   # save images as PNG files
)

print(f"[convert] Reading demos from {args.demos} ...")

with h5py.File(args.demos, "r") as h5:
    episodes = h5["episodes"]
    ep_keys = sorted(episodes.keys())
    print(f"  {len(ep_keys)} episodes found")

    for ep_idx, ep_key in enumerate(ep_keys):
        g = episodes[ep_key]
        front_arr = g["front_rgb"][:]   # (T, H, W, 3) uint8
        wrist_arr = g["wrist_rgb"][:]   # (T, H, W, 3) uint8
        state_arr = g["state"][:]       # (T, 8) float32
        action_arr = g["action"][:]     # (T, 7) float32
        T = len(action_arr)

        for t in range(T):
            frame = {
                "observation.images.front": PILImage.fromarray(front_arr[t]),
                "observation.images.wrist": PILImage.fromarray(wrist_arr[t]),
                "observation.state": state_arr[t],
                "action": action_arr[t],
                "task": args.task_desc,
            }
            ds.add_frame(frame)

        ds.save_episode()

        if (ep_idx + 1) % 50 == 0:
            print(f"  [{ep_idx+1}/{len(ep_keys)} episodes saved]")

print(f"[convert] {len(ep_keys)} episodes saved to {OUT_DIR}")

# Compute and save stats (required for openpi normalization)
print("[convert] Computing stats ...")
from lerobot.common.datasets.compute_stats import compute_stats
stats = compute_stats(ds)
ds.meta.save_stats(stats)

print(f"[convert] Done. Total frames: {ds.meta.total_frames}")
