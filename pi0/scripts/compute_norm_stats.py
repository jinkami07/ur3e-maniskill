"""
LeRobotデータセットからopenpi用norm_stats.jsonを計算するスクリプト

Usage:
  python scripts/compute_norm_stats.py
"""
import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, "/opt/openpi/src")

import os
os.environ["HF_LEROBOT_HOME"] = "/opt/pickcube_lerobot_v2"

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ASSETS_DIR = Path("/opt/pickcube_lerobot_v2/dataset/pickcube")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
ds = LeRobotDataset("dataset", root="/opt/pickcube_lerobot_v2/dataset")
print(f"  {len(ds)} frames loaded")

states  = []
actions = []

for i in range(len(ds)):
    sample = ds[i]
    states.append(np.array(sample["observation.state"], dtype=np.float32))
    actions.append(np.array(sample["action"], dtype=np.float32))

states  = np.stack(states)   # (N, 8)
actions = np.stack(actions)  # (N, 7)

# Truncate state to 7 dims (as done in TensorImagesToNumpy)
states = states[:, :7]

def compute_stats(arr):
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std":  arr.std(axis=0).clip(min=1e-6).tolist(),
        "q01":  np.percentile(arr, 1, axis=0).tolist(),
        "q99":  np.percentile(arr, 99, axis=0).tolist(),
    }

norm_stats = {
    "norm_stats": {
        "state":   compute_stats(states),
        "actions": compute_stats(actions),
    }
}

out_path = ASSETS_DIR / "norm_stats.json"
with open(out_path, "w") as f:
    json.dump(norm_stats, f, indent=2)

print(f"Saved norm_stats to {out_path}")
print(f"  state  mean: {np.array(norm_stats['norm_stats']['state']['mean']).round(4)}")
print(f"  action mean: {np.array(norm_stats['norm_stats']['actions']['mean']).round(4)}")
print("Done.")
