"""
PickCube-v1 pi0 LoRA 訓練スクリプト (wandb rich logging 付き)

openpi の built-in wandb サポート (wandb_enabled=True) で基本メトリクスを記録し、
訓練完了後に PickCube-v1 ロールアウト評価を実行して動画+成功率を wandb に追記する。

追加 wandb 機能:
  - wandb.config にハイパーパラメータを記録
  - 訓練完了後: PickCube-v1 ロールアウト × 5 エピソードで成功率 + 動画を記録

Usage (inside container):
  WANDB_API_KEY=<key> python scripts/train_pickcube.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

OPENPI_ROOT = Path("/opt/openpi")
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, "/workspace/src")

import numpy as np
import wandb

# ── wandb login ────────────────────────────────────────────────────────────────
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
else:
    print("[warn] WANDB_API_KEY not set — wandb may not work")

# ── openpi imports ─────────────────────────────────────────────────────────────
import openpi.training.train as _train
import openpi.training.config as _cfg
from pickcube_openpi_config import get_pickcube_config

CONFIG: _cfg.TrainConfig = get_pickcube_config()


# ── Post-training rollout eval ─────────────────────────────────────────────────

def run_rollout_eval(checkpoint_dir: str, num_episodes: int = 5, max_steps: int = 200):
    """
    Load the trained checkpoint and run PickCube-v1 rollout evaluation.
    Logs success_rate and rollout video to the current wandb run.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import jax

    print(f"[eval] Loading checkpoint from {checkpoint_dir}")

    # openpi inference server approach: use openpi.policies
    try:
        from openpi.policies import policy_config as pcfg
        from openpi import config as openpi_config
        # Build inference from checkpoint
        model_config = CONFIG.model
        policy = pcfg.create_trained_policy(
            model_config,
            checkpoint_dir=checkpoint_dir,
            repack_transforms=None,
        )
    except Exception as e:
        print(f"[eval] Could not load policy: {e}")
        print("[eval] Skipping rollout eval")
        return

    print(f"[eval] Running {num_episodes} episodes ...")
    successes = 0
    all_frames = []

    env = gym.make(
        "PickCube-v1",
        obs_mode="rgbd",
        robot_uids="panda",
        control_mode="pd_ee_delta_pose",
        sensor_configs={"width": 224, "height": 224},
        render_mode="rgb_array",
    )

    PROMPT = "pick up the red cube"

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        ep_frames = []
        done = False
        step = 0

        while not done and step < max_steps:
            # Build observation dict for policy
            sd = obs["sensor_data"]

            front = sd["base_camera"]["rgb"]
            if hasattr(front, "cpu"):
                front = front.cpu()
            front = np.asarray(front, dtype=np.uint8)
            if front.ndim == 4:
                front = front[0]

            if "hand_camera" in sd:
                wrist = sd["hand_camera"]["rgb"]
                if hasattr(wrist, "cpu"):
                    wrist = wrist.cpu()
                wrist = np.asarray(wrist, dtype=np.uint8)
                if wrist.ndim == 4:
                    wrist = wrist[0]
            else:
                wrist = front.copy()

            tcp = obs["extra"]["tcp_pose"]
            if hasattr(tcp, "cpu"):
                tcp = tcp.cpu()
            tcp = np.asarray(tcp, dtype=np.float32).reshape(-1)[:7]

            qpos = obs["agent"]["qpos"]
            if hasattr(qpos, "cpu"):
                qpos = qpos.cpu()
            qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
            gripper = np.array([qpos[-2:].mean()], dtype=np.float32)
            state = np.concatenate([tcp, gripper])  # (8,)

            policy_obs = {
                "images": {"front": front, "wrist": wrist},
                "state": state,
                "prompt": PROMPT,
            }

            try:
                action_chunk = policy.infer(policy_obs)
                action = np.asarray(action_chunk["actions"][0], dtype=np.float32)
            except Exception as e:
                print(f"[eval] inference error: {e}")
                break

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frame = env.render()
            if frame is not None:
                if hasattr(frame, "cpu"):
                    frame = frame.cpu()
                ep_frames.append(np.asarray(frame, dtype=np.uint8))

            step += 1

        success = bool(info.get("success", False))
        if success:
            successes += 1
        print(f"  ep {ep+1}/{num_episodes}: {'SUCCESS' if success else 'FAIL'} ({step} steps)")

        if ep_frames:
            all_frames.append(ep_frames)

    env.close()

    success_rate = successes / num_episodes
    print(f"[eval] Success rate: {success_rate:.2f} ({successes}/{num_episodes})")

    # Log to wandb
    log_dict = {"eval/success_rate": success_rate, "eval/num_episodes": num_episodes}

    # Log one video per episode (up to 5)
    for ep_idx, frames in enumerate(all_frames[:5]):
        if len(frames) < 2:
            continue
        video = np.stack(frames)             # (T, H, W, 3)
        video = video.transpose(0, 3, 1, 2)  # (T, C, H, W)
        log_dict[f"eval/rollout_ep{ep_idx}"] = wandb.Video(video, fps=10, format="mp4")

    wandb.log(log_dict)
    wandb.summary["eval/success_rate"] = success_rate


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    config = CONFIG
    t0 = time.time()

    print(f"[train] Config: {config.name}")
    print(f"[train] Steps: {config.num_train_steps}, batch: {config.batch_size}")
    print(f"[train] Checkpoint dir: {config.checkpoint_base_dir}")

    # openpi's built-in training (wandb_enabled=True handles basic metrics)
    try:
        _train.train(config)
    except Exception as e:
        print(f"[train] ERROR: {e}")
        raise

    elapsed = time.time() - t0
    print(f"[train] Finished in {elapsed/3600:.1f}h")

    # Post-training evaluation
    ckpt_dir = str(Path(config.checkpoint_base_dir) / config.name)
    # Find latest checkpoint
    ckpt_path = Path(ckpt_dir)
    if ckpt_path.exists():
        # openpi saves checkpoints as step_XXXXX directories
        ckpt_dirs = sorted(ckpt_path.glob("*/"), key=lambda p: p.name)
        if ckpt_dirs:
            latest = str(ckpt_dirs[-1])
            print(f"[eval] Latest checkpoint: {latest}")
            run_rollout_eval(latest, num_episodes=5, max_steps=200)

    print("[train] All done.")


if __name__ == "__main__":
    main()
