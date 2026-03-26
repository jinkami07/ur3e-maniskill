"""
Debug rollout: per-step logging of TCP pos, cube pos, actions, gripper
Usage:
  python pi0/scripts/debug_rollout.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path

OPENPI_ROOT = Path("/opt/openpi")
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, "/workspace/pi0/src")

import numpy as np
from pickcube_openpi_config import get_pickcube_config

CONFIG = get_pickcube_config()
CHECKPOINT = "/opt/checkpoints/pi0_pickcube_lora/lora_ft_v4/99999"

import gymnasium as gym
import mani_skill.envs  # noqa

# ── Check available cameras ────────────────────────────────────────────
env_probe = gym.make(
    "PickCube-v1",
    obs_mode="rgbd",
    robot_uids="panda",
    control_mode="pd_ee_delta_pose",
    sensor_configs={"width": 224, "height": 224},
)
obs_probe, _ = env_probe.reset(seed=0)
sd_probe = obs_probe["sensor_data"]
print("\n=== Available cameras ===")
for cam_name, cam_data in sd_probe.items():
    for key, val in cam_data.items():
        arr = np.asarray(val)
        print(f"  {cam_name}/{key}: shape={arr.shape}, dtype={arr.dtype}")
has_hand_camera = "hand_camera" in sd_probe
print(f"\nhand_camera available: {has_hand_camera}")
env_probe.close()

# ── Load policy ────────────────────────────────────────────────────────
print(f"\n[debug] Loading checkpoint from {CHECKPOINT}")
from openpi.policies import policy_config as pcfg
policy = pcfg.create_trained_policy(
    CONFIG,
    checkpoint_dir=CHECKPOINT,
    repack_transforms=None,
)
print("[debug] Policy loaded.")

PROMPT = "pick up the red cube"
PANDA_HOME_QPOS = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854, 0.04, 0.04], dtype=np.float32)

env = gym.make(
    "PickCube-v1",
    obs_mode="rgbd",
    robot_uids="panda",
    control_mode="pd_ee_delta_pose",
    sensor_configs={"width": 224, "height": 224},
    render_mode=None,
    max_episode_steps=300,
)

# Run 1 episode with full diagnostics (seed=0, same as training)
for ep_seed in [0, 1, 3000]:
    print(f"\n{'='*60}")
    print(f"Episode seed={ep_seed}")
    obs, _ = env.reset(seed=ep_seed)
    env_u = env.unwrapped
    env_u.agent.robot.set_qpos(PANDA_HOME_QPOS)
    env_u.agent.robot.set_qvel(np.zeros(9))
    try:
        env_u.goal_site.set_visibility(0)
    except Exception:
        pass

    def _np(t):
        if hasattr(t, "cpu"):
            t = t.cpu()
        return np.asarray(t)

    step = 0
    done = False
    MAX_STEPS = 50  # print first 50 steps for diagnosis
    PRINT_INTERVAL = 5

    while not done and step < MAX_STEPS:
        sd = obs["sensor_data"]
        front = _np(sd["base_camera"]["rgb"]).astype(np.uint8)
        if front.ndim == 4:
            front = front[0]
        if has_hand_camera:
            wrist = _np(sd["hand_camera"]["rgb"]).astype(np.uint8)
            if wrist.ndim == 4:
                wrist = wrist[0]
        else:
            wrist = front.copy()

        tcp7 = _np(obs["extra"]["tcp_pose"]).astype(np.float32).reshape(-1)[:7]
        qpos = _np(obs["agent"]["qpos"]).astype(np.float32).reshape(-1)
        gripper_w = qpos[-2:].mean()
        state = np.concatenate([tcp7, [gripper_w]])

        cube_pos = _np(env_u.cube.pose.p).reshape(-1)[:3]
        tcp_pos = tcp7[:3]
        dist = np.linalg.norm(tcp_pos - cube_pos)

        policy_obs = {
            "image": {"front": front, "wrist": wrist},
            "state": state,
            "prompt": PROMPT,
        }
        action_chunk = policy.infer(policy_obs)
        actions = np.asarray(action_chunk["actions"], dtype=np.float32)  # (8, 7)

        if step % PRINT_INTERVAL == 0:
            print(f"\n  step={step:3d} | tcp={tcp_pos.round(3)} | cube={cube_pos.round(3)} | dist={dist:.4f} | gripper_w={gripper_w:.4f}")
            print(f"           | action[0]={actions[0].round(4)}  (xyz={actions[0,:3].round(4)}, rot={actions[0,3:6].round(4)}, grip={actions[0,6]:.4f})")
            print(f"           | action mean xyz={actions[:4,:3].mean(axis=0).round(4)}, grip_mean={actions[:4,6].mean():.4f}")

        for action in actions[:4]:
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            if done:
                break

    print(f"\n  Final: step={step}, success={info.get('success', False)}")
    print(f"  Final TCP={tcp7[:3].round(4)}, cube={cube_pos.round(4)}, dist={dist:.4f}")

env.close()
print("\n[debug] Done.")
