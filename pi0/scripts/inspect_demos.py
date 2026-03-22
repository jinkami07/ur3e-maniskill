"""
収集済みデモの品質確認スクリプト

Usage:
  python scripts/inspect_demos.py --demos /opt/pickcube_demos/demos.h5
"""
import argparse
import numpy as np
import h5py
import imageio
import os

parser = argparse.ArgumentParser()
parser.add_argument("--demos", type=str, default="/opt/pickcube_demos/demos.h5")
parser.add_argument("--num-videos", type=int, default=5)
parser.add_argument("--out-dir", type=str, default="/workspace/output/demo_check")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

print(f"Loading demos from {args.demos} ...")

with h5py.File(args.demos, "r") as h5:
    episodes = h5["episodes"]
    ep_keys = sorted(episodes.keys())
    num_eps = len(ep_keys)
    print(f"\n=== Dataset Overview ===")
    print(f"Total episodes: {num_eps}")

    lengths = []
    action_mags = []
    gripper_vals = []
    state_ranges = []

    for ep_key in ep_keys:
        g = episodes[ep_key]
        action_arr = g["action"][:]   # (T, 7)
        state_arr  = g["state"][:]    # (T, 8)
        T = len(action_arr)
        lengths.append(T)
        action_mags.append(np.abs(action_arr[:, :6]).mean())
        gripper_vals.append(action_arr[:, 6])
        state_ranges.append(state_arr.max(axis=0) - state_arr.min(axis=0))

    lengths = np.array(lengths)
    print(f"\n=== Episode Lengths ===")
    print(f"  min: {lengths.min()}, max: {lengths.max()}, mean: {lengths.mean():.1f}")
    print(f"  Episodes < 10 steps (suspicious): {(lengths < 10).sum()}")
    print(f"  Episodes > 250 steps: {(lengths > 250).sum()}")

    print(f"\n=== Action Statistics ===")
    all_actions = np.concatenate([episodes[k]["action"][:] for k in ep_keys])
    print(f"  delta_pos (xyz) mean abs: {np.abs(all_actions[:, :3]).mean():.4f}")
    print(f"  delta_rot (rpy) mean abs: {np.abs(all_actions[:, 3:6]).mean():.4f}")
    print(f"  gripper mean: {all_actions[:, 6].mean():.4f}, range: [{all_actions[:, 6].min():.3f}, {all_actions[:, 6].max():.3f}]")

    print(f"\n=== State Statistics ===")
    all_states = np.concatenate([episodes[k]["state"][:] for k in ep_keys])
    print(f"  EEF pos range: {all_states[:, :3].min(axis=0).round(3)} ~ {all_states[:, :3].max(axis=0).round(3)}")
    print(f"  Gripper range: [{all_states[:, 7].min():.3f}, {all_states[:, 7].max():.3f}]")

    # Save demo videos
    print(f"\n=== Saving {args.num_videos} demo videos ===")
    for i, ep_key in enumerate(ep_keys[:args.num_videos]):
        g = episodes[ep_key]
        front = g["front_rgb"][:]  # (T, H, W, 3)
        T = len(front)
        out_path = f"{args.out_dir}/demo_{i:03d}_T{T}.mp4"
        imageio.mimsave(out_path, front, fps=10)
        print(f"  Saved: {out_path} ({T} frames)")

print(f"\n=== Done ===")
