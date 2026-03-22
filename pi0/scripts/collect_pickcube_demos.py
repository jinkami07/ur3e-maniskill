"""
PickCube-v1 デモ収集スクリプト (スクリプテッドポリシー版)

mplib/pinocchio を一切使わず、フェーズベースの EEF delta ポリシーで
成功エピソードを収集し HDF5 に保存する。

フェーズ:
  0: キューブ真上へ移動 (gripper open)
  1: キューブ高さへ降下 (gripper open)
  2: グリッパー閉じる (is_grasped まで待つ)
  3: goal_pos まで持ち上げる (gripper close)

Usage (inside container):
  python scripts/collect_pickcube_demos.py --num-demos 500 --out /opt/pickcube_demos/demos.h5
"""
import argparse
from pathlib import Path

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401  – task registration
import numpy as np
import torch

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num-demos",      type=int,   default=500)
parser.add_argument("--out",            type=str,   default="/opt/pickcube_demos/demos.h5")
parser.add_argument("--cam-size",       type=int,   default=224)
parser.add_argument("--seed-start",     type=int,   default=0)
parser.add_argument("--max-steps",      type=int,   default=300)
parser.add_argument("--approach-noise", type=float, default=0.02,
                    help="XY noise (m) added to approach target for diversity")
args = parser.parse_args()

rng = np.random.default_rng(seed=42)

OUT_PATH = Path(args.out)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
CAM = args.cam_size


# ── Helpers ───────────────────────────────────────────────────────────────────

def _np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


_SCALE = 10.0  # pd_ee_delta_pose: action=1.0 ≈ 0.1m delta (normalize to [-1,1])


def _delta_action(tcp_pos, target_pos, gripper):
    """
    Return 7-dim pd_ee_delta_pose action (normalized to [-1, 1]).
    _SCALE converts metres to normalized units: 1.0m * _SCALE → clipped to 1.0.
    tcp_pos: (3,) current EEF xyz
    target_pos: (3,) target xyz
    gripper: float scalar (1=open, -1=close, normalized)
    """
    delta = np.clip(_SCALE * (target_pos - tcp_pos), -1.0, 1.0)
    return np.concatenate([delta, np.zeros(3), [gripper]]).astype(np.float32)


# ── Episode collection ────────────────────────────────────────────────────────

def collect_episode(env, seed, xy_noise=0.0):
    """
    Run one episode of PickCube-v1 with phase-based scripted policy.
    Returns data dict on success, None on failure.
    """
    obs, _ = env.reset(seed=seed)
    env_u = env.unwrapped

    # Hide goal_site (green sphere) — visual noise unrelated to "pick up the red cube"
    try:
        env_u.goal_site.set_visibility(0)
    except Exception:
        pass

    fronts, wrists, states, acts = [], [], [], []

    phase = 0
    phase_timer = 0
    OPEN, CLOSE = 1.0, -1.0
    gripper_closed = False  # once closed, never reopen

    for step in range(args.max_steps):
        # ── Extract observation ──────────────────────────────────────────────
        sd = obs["sensor_data"]
        front = _np(sd["base_camera"]["rgb"])[0].astype(np.uint8)
        if "hand_camera" in sd:
            wrist = _np(sd["hand_camera"]["rgb"])[0].astype(np.uint8)
        else:
            wrist = front.copy()

        tcp7   = _np(obs["extra"]["tcp_pose"])[0].astype(np.float32)   # (7,) pos+quat
        qpos   = _np(obs["agent"]["qpos"])[0].astype(np.float32)       # (9,)
        gripper_w = qpos[-2:].mean().astype(np.float32)
        state  = np.concatenate([tcp7, [gripper_w]])                    # (8,)

        tcp_pos    = tcp7[:3]
        cube_pos   = _np(env_u.cube.pose.p)[0]
        goal_pos   = _np(env_u.goal_site.pose.p)[0]
        is_grasped = bool(_np(obs["extra"]["is_grasped"]).flat[0])

        # ── Phase logic ──────────────────────────────────────────────────────
        if phase == 0:
            # Move above cube with gripper open (with slight XY noise for diversity)
            above = cube_pos + np.array([xy_noise[0], xy_noise[1], 0.20])
            action = _delta_action(tcp_pos, above, OPEN)
            xy_dist = np.linalg.norm(tcp_pos[:2] - (cube_pos[:2] + xy_noise[:2]))
            if tcp_pos[2] > cube_pos[2] + 0.15 and xy_dist < 0.04:
                phase = 1

        elif phase == 1:
            # Descend onto cube (grasp height ≈ cube centre)
            grasp_target = cube_pos + np.array([0.0, 0.0, 0.01])
            action = _delta_action(tcp_pos, grasp_target, OPEN)
            dist = np.linalg.norm(tcp_pos - grasp_target)
            if dist < 0.015:
                phase = 2
                phase_timer = 0

        elif phase == 2:
            # Close gripper, hold position
            gripper_closed = True
            grasp_target = cube_pos + np.array([0.0, 0.0, 0.01])
            action = _delta_action(tcp_pos, grasp_target, CLOSE)
            phase_timer += 1
            if is_grasped or phase_timer >= 20:
                phase = 3

        else:
            # Lift to goal_site position
            action = _delta_action(tcp_pos, goal_pos, CLOSE)

        # Gripper lock: once closed, force CLOSE for rest of episode
        if gripper_closed:
            action = action.copy()
            action[6] = CLOSE

        # Record (obs BEFORE step)
        fronts.append(front)
        wrists.append(wrist)
        states.append(state)
        acts.append(action)

        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    success = bool(_np(info.get("success", False)).flat[0]
                   if hasattr(info.get("success", False), "shape")
                   else info.get("success", False))

    if not success or len(fronts) < 10:
        return None

    T = len(fronts)
    return {
        "front_rgb": np.stack(fronts).astype(np.uint8),    # (T, H, W, 3)
        "wrist_rgb": np.stack(wrists).astype(np.uint8),
        "state":     np.stack(states).astype(np.float32),  # (T, 8)
        "action":    np.stack(acts).astype(np.float32),    # (T, 7)
    }


# ── Main collection loop ──────────────────────────────────────────────────────

env = gym.make(
    "PickCube-v1",
    obs_mode="rgbd",
    robot_uids="panda",
    control_mode="pd_ee_delta_pose",
    sensor_configs={"width": CAM, "height": CAM},
    render_mode=None,
)

print(f"[collect] Collecting {args.num_demos} demos → {OUT_PATH}", flush=True)

successes = 0
failures  = 0
MAX_TRIES = args.num_demos * 8  # allow up to 8x attempts

with h5py.File(OUT_PATH, "w") as h5:
    ep_grp = h5.create_group("episodes")
    meta   = h5.create_group("meta")
    meta.attrs["cam_size"] = CAM
    meta.attrs["robot"]    = "panda"
    meta.attrs["task"]     = "PickCube-v1"
    meta.attrs["policy"]   = "scripted_eef_delta"

    for i in range(MAX_TRIES):
        if successes >= args.num_demos:
            break
        seed = args.seed_start + i
        xy_noise = rng.uniform(-args.approach_noise, args.approach_noise, size=2)
        data = collect_episode(env, seed, xy_noise=np.append(xy_noise, 0.0))

        if data is None:
            failures += 1
            if failures % 200 == 0:
                print(f"  [collect] {successes} ok / {failures} fail (seed={seed})",
                      flush=True)
            continue

        g = ep_grp.create_group(f"ep_{successes:05d}")
        for k, v in data.items():
            g.create_dataset(k, data=v, compression="gzip")
        g.attrs["seed"]      = seed
        g.attrs["num_steps"] = len(data["action"])
        successes += 1

        if successes % 50 == 0:
            print(f"  [collect] {successes}/{args.num_demos} collected "
                  f"(failures: {failures})", flush=True)

    meta.attrs["num_episodes"] = successes
    meta.attrs["num_failures"] = failures

env.close()
print(f"[collect] Done: {successes} episodes saved  ({failures} failures)", flush=True)
