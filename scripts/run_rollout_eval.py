"""
PickCube-v1 ロールアウト評価スクリプト (standalone)

Usage:
  python scripts/run_rollout_eval.py --checkpoint /opt/checkpoints/pi0_pickcube_lora/lora_ft_v1/9999
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-episodes", type=int, default=10)
parser.add_argument("--max-steps", type=int, default=200)
args = parser.parse_args()

OPENPI_ROOT = Path("/opt/openpi")
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, "/workspace/src")

import numpy as np
import wandb
from pickcube_openpi_config import get_pickcube_config

CONFIG = get_pickcube_config()

# Init wandb
wandb.init(
    project=CONFIG.project_name,
    name="rollout_eval_9999",
    resume="allow",
)

import gymnasium as gym
import mani_skill.envs  # noqa

print(f"[eval] Loading checkpoint from {args.checkpoint}")

try:
    from openpi.policies import policy_config as pcfg
    policy = pcfg.create_trained_policy(
        CONFIG,
        checkpoint_dir=args.checkpoint,
        repack_transforms=None,
    )
except Exception as e:
    print(f"[eval] Could not load policy: {e}")
    sys.exit(1)

print(f"[eval] Running {args.num_episodes} episodes ...")

env = gym.make(
    "PickCube-v1",
    obs_mode="rgbd",
    robot_uids="panda",
    control_mode="pd_ee_delta_pose",
    sensor_configs={"width": 224, "height": 224},
    render_mode="rgb_array",
    max_episode_steps=300,
)

PROMPT = "pick up the red cube"
successes = 0
all_frames = []

for ep in range(args.num_episodes):
    obs, _ = env.reset(seed=3000 + ep)
    ep_frames = []
    done = False
    step = 0
    info = {}

    while not done and step < args.max_steps:
        sd = obs["sensor_data"]

        def _np(t):
            if hasattr(t, "cpu"):
                t = t.cpu()
            return np.asarray(t)

        front = _np(sd["base_camera"]["rgb"]).astype(np.uint8)
        if front.ndim == 4:
            front = front[0]

        if "hand_camera" in sd:
            wrist = _np(sd["hand_camera"]["rgb"]).astype(np.uint8)
            if wrist.ndim == 4:
                wrist = wrist[0]
        else:
            wrist = front.copy()

        tcp = _np(obs["extra"]["tcp_pose"]).astype(np.float32).reshape(-1)[:7]
        qpos = _np(obs["agent"]["qpos"]).astype(np.float32).reshape(-1)
        gripper = np.array([qpos[-2:].mean()], dtype=np.float32)
        state = np.concatenate([tcp, gripper])  # (8,) — TensorImagesToNumpy truncates to 7

        policy_obs = {
            "image": {"front": front, "wrist": wrist},
            "state": state,
            "prompt": PROMPT,
        }

        try:
            action_chunk = policy.infer(policy_obs)
            actions = np.asarray(action_chunk["actions"], dtype=np.float32)  # (horizon, 7)
        except Exception as e:
            print(f"[eval] inference error: {e}")
            break

        # Execute 4 actions per inference call (action chunking)
        for action in actions[:4]:
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frame = env.render()
            if frame is not None:
                f = _np(frame).astype(np.uint8)
                if f.ndim == 4:
                    f = f[0]
                ep_frames.append(f)

            step += 1
            if done:
                break

    success = bool(info.get("success", False))
    if success:
        successes += 1
    print(f"  ep {ep+1}/{args.num_episodes}: {'SUCCESS' if success else 'FAIL'} ({step} steps)")

    if ep_frames:
        all_frames.append(ep_frames)

env.close()

success_rate = successes / args.num_episodes
print(f"[eval] Success rate: {success_rate:.2f} ({successes}/{args.num_episodes})")

log_dict = {"eval/success_rate": success_rate, "eval/num_episodes": args.num_episodes}
os.makedirs("/workspace/output/eval_videos", exist_ok=True)
for ep_idx, frames in enumerate(all_frames[:5]):
    if len(frames) < 2:
        continue
    video = np.stack(frames)          # (T, H, W, C)
    try:
        import imageio
        out_path = f"/workspace/output/eval_videos/ep{ep_idx}.mp4"
        imageio.mimsave(out_path, video, fps=10)
        print(f"[eval] Saved video: {out_path}")
        # Upload mp4 file directly (no moviepy needed)
        log_dict[f"eval/rollout_ep{ep_idx}"] = wandb.Video(out_path, fps=10, format="mp4")
    except Exception as e:
        print(f"[eval] video save/upload failed: {e}")

wandb.log(log_dict)
wandb.summary["eval/success_rate"] = success_rate
wandb.finish()
print("[eval] Done.")
