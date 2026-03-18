"""
HDF5 デモ → LeRobot HuggingFace データセット変換スクリプト

openpi は LeRobot 形式を期待するため、以下の列を持つ datasets.Dataset を作成する:
  observation.images.front  (uint8, H x W x 3)
  observation.images.wrist  (uint8, H x W x 3)
  observation.state         (float32, 8)
  action                    (float32, 7)
  episode_index             (int64)
  frame_index               (int64)
  timestamp                 (float64)
  task_description          (str)

Usage (inside container):
  python scripts/convert_demos_to_lerobot.py \
    --demos /opt/pickcube_demos/demos.h5 \
    --out   /opt/pickcube_lerobot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

parser = argparse.ArgumentParser()
parser.add_argument("--demos", type=str, default="/opt/pickcube_demos/demos.h5")
parser.add_argument("--out", type=str, default="/opt/pickcube_lerobot")
parser.add_argument("--fps", type=float, default=10.0)
parser.add_argument("--task-desc", type=str, default="pick up the red cube")
args = parser.parse_args()

OUT_DIR = Path(args.out)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[convert] Reading demos from {args.demos} ...")

rows = []

with h5py.File(args.demos, "r") as h5:
    episodes = h5["episodes"]
    ep_keys = sorted(episodes.keys())
    print(f"  {len(ep_keys)} episodes found")

    for ep_idx, ep_key in enumerate(ep_keys):
        g = episodes[ep_key]
        front = g["front_rgb"][:]   # (T, H, W, 3) uint8
        wrist = g["wrist_rgb"][:]   # (T, H, W, 3) uint8
        state = g["state"][:]       # (T, 8) float32
        action = g["action"][:]     # (T, 7) float32
        T = len(action)

        for t in range(T):
            rows.append({
                "observation.images.front": front[t],
                "observation.images.wrist": wrist[t],
                "observation.state": state[t].tolist(),
                "action": action[t].tolist(),
                "episode_index": ep_idx,
                "frame_index": t,
                "timestamp": float(t) / args.fps,
                "task_description": args.task_desc,
            })

        if ep_idx % 50 == 0:
            print(f"  processed {ep_idx}/{len(ep_keys)} episodes ({len(rows)} frames)")

print(f"[convert] Total frames: {len(rows)}")

# ── Build HuggingFace Dataset ─────────────────────────────────────────────────
# images are stored as PIL Images (HF Image feature)

print("[convert] Building HuggingFace Dataset ...")

def make_pil(arr):
    return PILImage.fromarray(arr.astype(np.uint8))

# Separate columns
fronts = [make_pil(r["observation.images.front"]) for r in rows]
wrists = [make_pil(r["observation.images.wrist"]) for r in rows]
states = [r["observation.state"] for r in rows]
actions_list = [r["action"] for r in rows]
ep_idxs = [r["episode_index"] for r in rows]
frame_idxs = [r["frame_index"] for r in rows]
timestamps = [r["timestamp"] for r in rows]
task_descs = [r["task_description"] for r in rows]

features = Features({
    "observation.images.front": Image(),
    "observation.images.wrist": Image(),
    "observation.state": Sequence(Value("float32"), length=8),
    "action": Sequence(Value("float32"), length=7),
    "episode_index": Value("int64"),
    "frame_index": Value("int64"),
    "timestamp": Value("float64"),
    "task_description": Value("string"),
})

ds = Dataset.from_dict(
    {
        "observation.images.front": fronts,
        "observation.images.wrist": wrists,
        "observation.state": states,
        "action": actions_list,
        "episode_index": ep_idxs,
        "frame_index": frame_idxs,
        "timestamp": timestamps,
        "task_description": task_descs,
    },
    features=features,
)

# ── Save ──────────────────────────────────────────────────────────────────────
ds.save_to_disk(str(OUT_DIR))
print(f"[convert] Saved dataset ({len(ds)} rows) to {OUT_DIR}")

# ── Compute norm stats ────────────────────────────────────────────────────────
print("[convert] Computing norm stats ...")
states_np = np.array(states, dtype=np.float32)
actions_np = np.array(actions_list, dtype=np.float32)

norm_stats = {
    "state": {
        "mean": states_np.mean(0).tolist(),
        "std": states_np.std(0).clip(1e-6).tolist(),
        "min": states_np.min(0).tolist(),
        "max": states_np.max(0).tolist(),
        "q01": np.quantile(states_np, 0.01, axis=0).tolist(),
        "q99": np.quantile(states_np, 0.99, axis=0).tolist(),
    },
    "action": {
        "mean": actions_np.mean(0).tolist(),
        "std": actions_np.std(0).clip(1e-6).tolist(),
        "min": actions_np.min(0).tolist(),
        "max": actions_np.max(0).tolist(),
        "q01": np.quantile(actions_np, 0.01, axis=0).tolist(),
        "q99": np.quantile(actions_np, 0.99, axis=0).tolist(),
    },
}

import json
norm_stats_path = OUT_DIR / "norm_stats.json"
with open(norm_stats_path, "w") as f:
    json.dump(norm_stats, f, indent=2)
print(f"[convert] Norm stats saved to {norm_stats_path}")
print("[convert] Done.")
