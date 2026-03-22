"""
デモ動画保存スクリプト

Usage:
  python scripts/save_demo_videos.py --demos /opt/pickcube_demos/demos_v2.h5 --num 10 --out /opt/demo_videos
"""
import argparse
import os
import numpy as np
import h5py
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("--demos", type=str, default="/opt/pickcube_demos/demos_v2.h5")
parser.add_argument("--num", type=int, default=10)
parser.add_argument("--out", type=str, default="/opt/demo_videos")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

with h5py.File(args.demos, "r") as h5:
    episodes = h5["episodes"]
    ep_keys = sorted(episodes.keys())[:args.num]
    print(f"Saving {len(ep_keys)} demo videos to {args.out}")

    for i, key in enumerate(ep_keys):
        g = episodes[key]
        front = g["front_rgb"][:]   # (T, H, W, 3)
        action = g["action"][:]     # (T, 7)
        T = len(front)
        gripper = action[:, 6]

        # Annotate gripper state in title
        out_path = f"{args.out}/demo_{i:03d}_T{T}.mp4"
        imageio.mimsave(out_path, front, fps=10)
        print(f"  [{i}] {key}: {T} steps, gripper min={gripper.min():.2f} max={gripper.max():.2f} → {out_path}")

print("Done.")
