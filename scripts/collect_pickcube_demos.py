"""
PickCube-v1 デモ収集スクリプト
Panda の Motion Planning ソルバーでデモを生成し、
RGB観測 + TCP pose + グリッパー幅 + delta EEF action を HDF5 に保存する。

Usage (inside container):
  python scripts/collect_pickcube_demos.py --num-demos 500 --out /opt/pickcube_demos/demos.h5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401 – task registration
import numpy as np
import sapien
import torch
from mani_skill.examples.motionplanning.panda.solutions.pick_cube import solve
from mani_skill.utils.wrappers.record import RecordEpisode

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num-demos", type=int, default=500)
parser.add_argument("--out", type=str, default="/opt/pickcube_demos/demos.h5")
parser.add_argument("--cam-size", type=int, default=224)
parser.add_argument("--seed-start", type=int, default=0)
args = parser.parse_args()

OUT_PATH = Path(args.out)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
CAM = args.cam_size

# ── Env ───────────────────────────────────────────────────────────────────────
env = gym.make(
    "PickCube-v1",
    obs_mode="rgbd",
    robot_uids="panda",
    control_mode="pd_ee_delta_pose",   # we need EEF delta actions
    sensor_configs={"width": CAM, "height": CAM},
    render_mode=None,
)


def _to_np(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    return np.asarray(t)


def collect_episode(seed: int):
    """
    Returns dict of arrays for one successful episode, or None on failure.
    Arrays: front_rgb (T,H,W,3), wrist_rgb (T,H,W,3),
            state (T,8), actions (T,7), success bool.
    """
    obs, _ = env.reset(seed=seed)

    # ── Reset env in pd_joint_pos mode first, then re-init with pd_ee_delta_pose
    # Actually we keep pd_ee_delta_pose throughout. Motion planner generates
    # joint-space trajectories internally but we capture EEF observations.
    front_rgbs, wrist_rgbs, states, actions = [], [], [], []
    last_tcp_pose = None

    def _obs_snapshot(o):
        """Extract front RGB, state (eef_pos+quat+gripper)."""
        sd = o["sensor_data"]
        front = _to_np(sd["base_camera"]["rgb"])[0]  # (H,W,3) uint8

        # wrist camera if available, else copy front
        if "hand_camera" in sd:
            wrist = _to_np(sd["hand_camera"]["rgb"])[0]
        else:
            wrist = front.copy()

        tcp = _to_np(o["extra"]["tcp_pose"])[0]         # (7,) pos+quat
        qpos = _to_np(o["agent"]["qpos"])[0]            # (9,) for panda
        gripper_width = qpos[-2:].mean().reshape(1)     # avg finger width → (1,)
        state = np.concatenate([tcp, gripper_width]).astype(np.float32)  # (8,)
        return front, wrist, state, tcp

    # --- run motion planner (uses pd_joint_pos internally) ---
    # We use a temporary env in pd_joint_pos for planning, then replay in pd_ee_delta_pose
    # Simplest approach: create a separate planning env
    plan_env = gym.make(
        "PickCube-v1",
        obs_mode="none",
        robot_uids="panda",
        control_mode="pd_joint_pos",
        render_mode=None,
    )
    plan_env.reset(seed=seed)
    try:
        res = solve(plan_env, seed=seed, debug=False, vis=False)
    except Exception as e:
        plan_env.close()
        return None
    plan_env.close()

    if res is None or res == -1:
        return None
    # res is list of (obs, reward, terminated, truncated, info) tuples
    # Just check last step success
    if not res[-1][-1].get("success", False):
        return None

    # --- Now replay with pd_ee_delta_pose to get EEF observations ---
    obs, _ = env.reset(seed=seed)
    f0, w0, s0, tcp0 = _obs_snapshot(obs)

    for step_result in res:
        # step_result = (obs_plan, reward, term, trunc, info)
        # We don't use the planned action directly; instead we derive delta EEF from TCP
        # by stepping the env with zeros and reading TCP, then computing delta
        # Since motion planner uses pd_joint_pos, we replay the joint positions
        # Here we take a simpler approach: just step with zero action and use TCP diff
        # Actually: use the motion planner's joint actions to replicate movement,
        # then compute EEF delta from consecutive TCP poses.
        break  # This approach is too complex; use direct obs collection below

    # ── Simpler approach: replay with pd_ee_delta_pose using computed deltas ──
    # Reset both envs with same seed and step the planning env, recording TCP.
    plan_env2 = gym.make(
        "PickCube-v1",
        obs_mode="rgbd",
        robot_uids="panda",
        control_mode="pd_joint_pos",
        sensor_configs={"width": CAM, "height": CAM},
        render_mode=None,
    )
    obs_p, _ = plan_env2.reset(seed=seed)
    try:
        res2 = solve(plan_env2, seed=seed, debug=False, vis=False)
    except Exception:
        plan_env2.close()
        return None
    plan_env2.close()

    if res2 is None or res2 == -1:
        return None

    # res2 is list of step results; we need to rebuild the full trajectory
    # Actually solve() returns only the last step result. We need to collect per-step.
    # Let's use a wrapper approach instead.
    return None  # Will be replaced by the wrapper approach below


# ── Better approach: monkey-patch the env to record steps ─────────────────────

class StepRecorder:
    """Wraps gym env to record observations and EEF deltas at each step."""

    def __init__(self, cam_size: int):
        self.env = gym.make(
            "PickCube-v1",
            obs_mode="rgbd",
            robot_uids="panda",
            control_mode="pd_joint_pos",
            sensor_configs={"width": cam_size, "height": cam_size},
            render_mode=None,
        )
        self.reset_episode()

    def reset_episode(self):
        self._front, self._wrist, self._states = [], [], []
        self._tcp_poses: list[np.ndarray] = []

    def reset(self, seed):
        obs, info = self.env.reset(seed=seed)
        self.reset_episode()
        self._record_obs(obs)
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        self._record_obs(out[0])
        return out

    def _record_obs(self, obs):
        sd = obs["sensor_data"]
        front = _to_np(sd["base_camera"]["rgb"])[0].astype(np.uint8)
        if "hand_camera" in sd:
            wrist = _to_np(sd["hand_camera"]["rgb"])[0].astype(np.uint8)
        else:
            wrist = front.copy()

        tcp = _to_np(obs["extra"]["tcp_pose"])[0].astype(np.float32)  # (7,)
        qpos = _to_np(obs["agent"]["qpos"])[0].astype(np.float32)
        gripper = np.array([qpos[-2:].mean()], dtype=np.float32)
        state = np.concatenate([tcp, gripper])  # (8,)

        self._front.append(front)
        self._wrist.append(wrist)
        self._states.append(state)
        self._tcp_poses.append(tcp)

    def get_episode_data(self):
        """Compute delta EEF actions from consecutive TCP poses."""
        T = len(self._front)
        if T < 2:
            return None

        fronts = np.stack(self._front[:-1])    # (T-1, H, W, 3)
        wrists = np.stack(self._wrist[:-1])    # (T-1, H, W, 3)
        states = np.stack(self._states[:-1])   # (T-1, 8)

        # Delta EEF: pos diff + rotation diff (axis-angle approx) + gripper
        acts = []
        for t in range(T - 1):
            curr = self._tcp_poses[t]
            nxt = self._tcp_poses[t + 1]
            delta_pos = (nxt[:3] - curr[:3]).astype(np.float32)

            # Rotation delta as axis-angle (simplified: small angle approx → quat diff)
            # For pi0, we just use delta pos + gripper (3+1=4 dim is also valid)
            # but pi0 expects 7-dim, so we include rotation delta as euler
            from scipy.spatial.transform import Rotation as R
            q_curr = curr[3:7]  # wxyz
            q_nxt = nxt[3:7]
            # Convert to scipy convention (xyzw)
            r_curr = R.from_quat([q_curr[1], q_curr[2], q_curr[3], q_curr[0]])
            r_nxt = R.from_quat([q_nxt[1], q_nxt[2], q_nxt[3], q_nxt[0]])
            r_delta = r_nxt * r_curr.inv()
            euler_delta = r_delta.as_euler("xyz").astype(np.float32)  # (3,)

            # Gripper: use next state's gripper width (normalized to [0,1])
            gripper_nxt = self._states[t + 1][-1]
            # Panda max finger width ≈ 0.04 m per finger
            gripper_norm = np.clip(gripper_nxt / 0.04, 0.0, 1.0).astype(np.float32)

            action = np.concatenate([delta_pos, euler_delta, [gripper_norm]])  # (7,)
            acts.append(action)

        actions = np.stack(acts).astype(np.float32)  # (T-1, 7)
        return {
            "front_rgb": fronts,
            "wrist_rgb": wrists,
            "state": states,
            "action": actions,
        }

    def close(self):
        self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped


# Monkey-patch solve to use our recorder
from mani_skill.examples.motionplanning.panda.solutions.pick_cube import solve as _orig_solve


def solve_and_record(recorder: StepRecorder, seed: int):
    """Run motion planning solution and record all steps."""

    # Patch the recorder's env step to intercept
    env_inner = recorder.env
    orig_step = env_inner.step

    def patched_step(action):
        obs, rew, term, trunc, info = orig_step(action)
        recorder._record_obs(obs)
        return obs, rew, term, trunc, info

    # Override env step inside planner by using unwrapped env directly
    # We'll use a different approach: run the planner with our env directly
    planner_env = recorder.env  # planner will call env.reset() and env.step()
    recorder.reset_episode()
    obs, _ = planner_env.reset(seed=seed)
    recorder._record_obs(obs)

    # Temporarily override step
    planner_env.step = patched_step
    try:
        res = _orig_solve(planner_env, seed=seed, debug=False, vis=False)
    except Exception as e:
        planner_env.step = orig_step
        return None
    planner_env.step = orig_step

    if res is None or res == -1:
        return None
    last_info = res[-1][-1] if isinstance(res, list) else res[-1]
    if not bool(last_info.get("success", False)):
        return None
    return recorder.get_episode_data()


# ── Main collection loop ────────────────────────────────────────────────────────

env.close()  # close the initial test env

recorder = StepRecorder(CAM)
print(f"[collect] Collecting {args.num_demos} demos → {OUT_PATH}")

successes = 0
failures = 0
all_episodes = []

with h5py.File(OUT_PATH, "w") as h5:
    ep_grp = h5.create_group("episodes")
    meta = h5.create_group("meta")
    meta.attrs["cam_size"] = CAM
    meta.attrs["robot"] = "panda"
    meta.attrs["task"] = "PickCube-v1"

    for i in range(args.num_demos * 3):  # try 3x to get enough successes
        if successes >= args.num_demos:
            break
        seed = args.seed_start + i
        data = solve_and_record(recorder, seed)
        if data is None:
            failures += 1
            continue

        g = ep_grp.create_group(f"ep_{successes:05d}")
        for k, v in data.items():
            g.create_dataset(k, data=v, compression="gzip")
        g.attrs["seed"] = seed
        g.attrs["num_steps"] = len(data["action"])
        successes += 1

        if successes % 50 == 0:
            print(f"  collected {successes}/{args.num_demos}  (failures: {failures})")

    meta.attrs["num_episodes"] = successes
    meta.attrs["num_failures"] = failures

recorder.close()
print(f"[collect] Done: {successes} episodes saved to {OUT_PATH}")
print(f"          {failures} failures skipped")
