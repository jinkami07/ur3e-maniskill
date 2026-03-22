"""
ManiSkill3 + UR3e 動作確認スクリプト

確認項目:
  1. mani_skill インポート
  2. UR3e エージェント登録
  3. PickCube-v1 環境の起動（UR3e）
  4. reset → ランダムアクション 10 ステップ → 観測形状を表示
  5. カメラ画像を output/smoke_test_frame.png に保存

Usage:
  python pi0/src/01_smoke_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")  # ur3e_agent (共有)

import numpy as np

# ── 1. ManiSkill3 インポート ──────────────────────────────────────────────────
try:
    import mani_skill
    import gymnasium as gym
    import mani_skill.envs  # タスク一式を登録
    print(f"[OK] mani_skill version: {mani_skill.__version__}")
except ImportError as e:
    raise SystemExit(
        f"\n[ERROR] mani_skill not installed: {e}\n"
        "  pip install mani-skill"
    )

# ── 2. UR3e エージェント登録 ──────────────────────────────────────────────────
try:
    import ur3e_agent  # noqa: F401  (register_agent デコレータが実行される)
    print(f"[OK] UR3e agent registered (URDF: {ur3e_agent.UR3E_URDF})")
except Exception as e:
    print(f"[WARN] UR3e agent registration failed: {e}")
    print("       → Falling back to 'panda' for smoke test")
    ROBOT_UID = "panda"
else:
    ROBOT_UID = "ur3e"

# ── 3. 環境起動 ───────────────────────────────────────────────────────────────
print(f"\n[info] Creating PickCube-v1 with robot={ROBOT_UID} ...")

try:
    env = gym.make(
        "PickCube-v1",
        obs_mode="rgbd",
        robot_uids=ROBOT_UID,
        sensor_configs={"width": 224, "height": 224},
        render_mode="rgb_array",
        num_envs=1,
    )
    print("[OK] Environment created")
except Exception as e:
    raise SystemExit(f"\n[ERROR] Failed to create environment: {e}")

# ── 4. reset + ランダムステップ ───────────────────────────────────────────────
obs, info = env.reset(seed=42)

print("\n[info] Observation structure:")
def _print_obs(d: dict | np.ndarray, prefix: str = "  ") -> None:
    if isinstance(d, dict):
        for k, v in d.items():
            _print_obs(v, prefix + f"{k}/")
    else:
        try:
            arr = np.asarray(d)
            print(f"{prefix[:-1]}: shape={arr.shape} dtype={arr.dtype}")
        except Exception:
            print(f"{prefix[:-1]}: {type(d)}")

_print_obs(obs)

print("\n[info] Running 10 random steps ...")
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  step {i+1:2d}  reward={float(reward):.4f}  done={bool(terminated or truncated)}")

# ── 5. カメラ画像を保存 ───────────────────────────────────────────────────────
out_dir = Path(os.environ.get("OUT_DIR", "output"))
out_dir.mkdir(parents=True, exist_ok=True)

try:
    import imageio

    # ManiSkill3 は obs の中に camera obs を持つ
    rgb = obs["sensor_data"]["base_camera"]["rgb"]
    # バッチ次元を除去 (1, H, W, 3) → (H, W, 3)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu()
    rgb = np.asarray(rgb, dtype=np.uint8)

    frame_path = out_dir / "smoke_test_frame.png"
    imageio.imwrite(str(frame_path), rgb)
    print(f"\n[saved] {frame_path}")
except Exception as e:
    print(f"\n[WARN] Failed to save frame: {e}")

env.close()
print("\n[OK] Smoke test PASSED")
