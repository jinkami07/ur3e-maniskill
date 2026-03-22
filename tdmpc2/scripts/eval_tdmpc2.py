"""
TD-MPC2 評価スクリプト

Usage:
  python tdmpc2/scripts/eval_tdmpc2.py --checkpoint /opt/checkpoints/tdmpc2/step_00100000.pt
"""
import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tdmpc2/src"))

from ur3e_pickcube_env import UR3ePickCubeEnv

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-episodes", type=int, default=20)
parser.add_argument("--save-video", action="store_true")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[eval] Loading checkpoint: {args.checkpoint}")

ckpt = torch.load(args.checkpoint, map_location=DEVICE)
step = ckpt.get("step", "?")
print(f"[eval] Checkpoint step: {step}")

# モデル復元
from train_tdmpc2 import SimpleTDMPC2
OBS_DIM, ACT_DIM = 34, 7
model = SimpleTDMPC2(OBS_DIM, ACT_DIM).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

env = UR3ePickCubeEnv({"render_mode": "rgb_array" if args.save_video else None})
successes = 0
frames_all = []

for ep in range(args.num_episodes):
    obs, _ = env.reset(seed=5000 + ep)
    done = False
    t = 0
    ep_frames = []
    while not done and t < 200:
        o_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            z = model.encode(o_t)
            a, _ = model.pi(z)
        obs, _, terminated, truncated, info = env.step(a.cpu().numpy().reshape(-1))
        done = terminated or truncated
        t += 1
        if args.save_video:
            f = env.render()
            if f is not None:
                ep_frames.append(f)

    ok = bool(info.get("success", False))
    successes += ok
    print(f"  ep {ep+1:>2d}: {'SUCCESS' if ok else 'FAIL  '} ({t} steps)")

    if args.save_video and ep_frames:
        frames_all.append(ep_frames)

env.close()
rate = successes / args.num_episodes
print(f"\n[eval] Success rate: {rate:.2f} ({successes}/{args.num_episodes})")

if args.save_video:
    import imageio
    out_dir = Path("tdmpc2/output")
    out_dir.mkdir(exist_ok=True)
    for i, frames in enumerate(frames_all[:5]):
        if len(frames) > 1:
            path = out_dir / f"ep{i:02d}_{'ok' if i < successes else 'fail'}.mp4"
            imageio.mimsave(str(path), np.stack(frames), fps=15)
            print(f"  Video saved: {path}")
