"""
ManiSkill3 + UR3e + pi0 評価スクリプト

- タスク: PickCube-v1 (キューブをピックして目標位置へ)
- ロボット: UR3e (6DOF arm + parallel gripper)
- カメラ: base_camera (front) + hand_camera (wrist) @ 224x224
- ポリシー: pi0 / pi05 (openpi)
- アクション: EEF delta [dx, dy, dz, droll, dpitch, dyaw, gripper] (7-dim)

Usage:
  # プリトレーニド (pi05)
  python src/02_ur3e_pi0_eval.py

  # ファインチューニング済みチェックポイント
  CKPT_DIR=/opt/openpi/checkpoints/my_checkpoint python src/02_ur3e_pi0_eval.py

  # 設定変更
  NUM_EPISODES=2 TOTAL_STEPS=100 python src/02_ur3e_pi0_eval.py
"""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path

import imageio
import numpy as np

# インタラクティブプロンプトを抑制
builtins.input = lambda *_, **__: "N"

# ── 設定（環境変数で上書き可） ────────────────────────────────────────────────
def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))

def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)

OPENPI_CONF  = _str("OPENPI_CONF", "pi05_libero")
CKPT_DIR     = _str("CKPT_DIR", "")

NUM_EPISODES = _int("NUM_EPISODES", 5)
TOTAL_STEPS  = _int("TOTAL_STEPS", 200)
HORIZON      = _int("HORIZON", 10)

CAM_H = _int("CAMERA_HEIGHT", 224)
CAM_W = _int("CAMERA_WIDTH",  224)

OUT_DIR = Path(_str("OUT_DIR", "output"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "pick up the red cube"

# ── ManiSkill3 + UR3e ─────────────────────────────────────────────────────────
try:
    import gymnasium as gym
    import mani_skill.envs  # タスク登録
except ImportError as e:
    raise SystemExit(f"[ERROR] mani_skill not installed: {e}")

try:
    import ur3e_agent  # UR3e を register_agent で登録
    ROBOT_UID = "ur3e"
    print(f"[info] UR3e agent registered (URDF: {ur3e_agent.UR3E_URDF})")
except Exception as e:
    print(f"[WARN] UR3e registration failed: {e}")
    print("       Using 'panda' as fallback. Set UR3E_URDF if URDF is missing.")
    ROBOT_UID = "panda"

# ── openpi ────────────────────────────────────────────────────────────────────
from openpi.shared import download
from openpi.policies import policy_config
from openpi.training import config as cfg

conf = cfg.get_config(OPENPI_CONF)

if CKPT_DIR:
    ckpt = CKPT_DIR
    print(f"[policy] Fine-tuned checkpoint: {ckpt}")
else:
    ckpt = download.maybe_download(f"gs://openpi-assets/checkpoints/{OPENPI_CONF}")
    print(f"[policy] Pretrained checkpoint: {OPENPI_CONF}")

policy = policy_config.create_trained_policy(conf, ckpt)

# ── 環境 ──────────────────────────────────────────────────────────────────────
def make_env():
    return gym.make(
        "PickCube-v1",
        obs_mode="rgbd",
        robot_uids=ROBOT_UID,
        sensor_configs={"width": CAM_H, "height": CAM_W},
        render_mode="rgb_array",
        num_envs=1,
    )

# ── 観測から必要な情報を取り出す ──────────────────────────────────────────────

def extract_images(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """base_camera (front) と hand_camera (wrist) の RGB を 224x224 uint8 で返す。"""
    def _get_rgb(sensor_key: str) -> np.ndarray:
        rgb = obs["sensor_data"][sensor_key]["rgb"]
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu()
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.ndim == 4:
            arr = arr[0]  # バッチ次元を除去
        return arr  # (H, W, 3)

    front_img = _get_rgb("base_camera")

    # wrist カメラがない場合は front を流用
    if "hand_camera" in obs["sensor_data"]:
        wrist_img = _get_rgb("hand_camera")
    else:
        wrist_img = front_img.copy()

    return front_img, wrist_img


def extract_state8(obs: dict) -> np.ndarray:
    """8-dim 固有感覚状態 [eef_pos(3), eef_quat(4), gripper(1)] を返す。"""
    agent_obs = obs.get("agent", {})

    # EEF pose は extra に入っていることが多い
    extra = obs.get("extra", {})

    eef_pos = np.zeros(3, dtype=np.float32)
    eef_quat = np.array([1, 0, 0, 0], dtype=np.float32)  # wxyz

    if "tcp_pose" in extra:
        tcp = np.asarray(extra["tcp_pose"]).ravel()
        if tcp.ndim == 2:
            tcp = tcp[0]
        eef_pos  = tcp[:3].astype(np.float32)
        eef_quat = tcp[3:7].astype(np.float32)
    elif "ee_pos" in extra:
        eef_pos = np.asarray(extra["ee_pos"]).ravel()[:3].astype(np.float32)

    # グリッパー状態: qpos の末尾
    qpos = np.asarray(agent_obs.get("qpos", np.zeros(7))).ravel()
    gripper_val = qpos[-1:].astype(np.float32)  # (1,)

    state = np.concatenate([eef_pos, eef_quat, gripper_val])
    out = np.zeros(8, dtype=np.float32)
    out[:min(8, state.size)] = state[:min(8, state.size)]
    return out

# ── エピソード実行 ────────────────────────────────────────────────────────────

def run_episode(env) -> tuple[bool, list[np.ndarray]]:
    obs, _ = env.reset()

    frames: list[np.ndarray] = []
    success = False

    for _chunk_start in range(0, TOTAL_STEPS, HORIZON):
        front_img, wrist_img = extract_images(obs)
        state8 = extract_state8(obs)

        sample = {
            "observation/state":       state8,
            "observation/image":       front_img,
            "observation/wrist_image": wrist_img,
            "prompt":                  PROMPT,
        }

        # pi0 推論 → アクションチャンク (HORIZON, 7)
        actions = policy.infer(sample)["actions"][:HORIZON]

        for a in actions:
            a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
            # ManiSkill3 は (num_envs, action_dim) を期待することがある
            action_input = a[np.newaxis] if ROBOT_UID != "panda" else a

            obs, reward, terminated, truncated, info = env.step(action_input)

            # フレーム保存
            try:
                front_img_step, _ = extract_images(obs)
                frames.append(front_img_step)
            except Exception:
                pass

            # 成功判定
            if float(reward) > 0:
                success = True
            if "success" in info:
                if bool(np.asarray(info["success"]).any()):
                    success = True

            if success or bool(terminated) or bool(truncated):
                return success, frames

    return success, frames

# ── メイン評価ループ ──────────────────────────────────────────────────────────

print("=" * 60)
print(f" UR3e + pi0  PickCube-v1 evaluation")
print(f" Config:    {OPENPI_CONF}  ckpt={CKPT_DIR or '(pretrained)'}")
print(f" Robot:     {ROBOT_UID}")
print(f" Episodes:  {NUM_EPISODES}  steps/ep={TOTAL_STEPS}  horizon={HORIZON}")
print(f" Camera:    {CAM_H}x{CAM_W}  output → {OUT_DIR}/")
print("=" * 60)

env = make_env()
results: list[dict] = []

for ep in range(NUM_EPISODES):
    print(f"\n[episode {ep + 1}/{NUM_EPISODES}]")
    success, frames = run_episode(env)

    tag = "success" if success else "failure"
    vpath = OUT_DIR / f"ep{ep:02d}_ur3e_pickcube_{tag}.mp4"

    if frames:
        with imageio.get_writer(str(vpath), fps=24, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)
        print(f"  [saved] {vpath}  ({len(frames)} frames)")

    results.append({"episode": ep, "success": success, "frames": len(frames)})
    print(f"  → {'SUCCESS' if success else 'failure'}  ({len(frames)} frames)")

env.close()

# ── サマリー ─────────────────────────────────────────────────────────────────
n_success = sum(r["success"] for r in results)
success_rate = n_success / NUM_EPISODES

print("\n" + "=" * 60)
print(f" Summary: {n_success}/{NUM_EPISODES} succeeded  ({success_rate:.1%})")
print("=" * 60)
print(f"  {'Ep':>4}  {'Result':<10}  {'Frames':>6}")
print(f"  {'-'*4}  {'-'*10}  {'-'*6}")
for r in results:
    tag = "SUCCESS" if r["success"] else "failure"
    print(f"  {r['episode']:>4}  {tag:<10}  {r['frames']:>6}")

summary = {
    "config":       OPENPI_CONF,
    "ckpt":         CKPT_DIR,
    "robot":        ROBOT_UID,
    "task":         "PickCube-v1",
    "prompt":       PROMPT,
    "num_episodes": NUM_EPISODES,
    "total_steps":  TOTAL_STEPS,
    "horizon":      HORIZON,
    "success_rate": success_rate,
    "n_success":    n_success,
    "episodes":     results,
}
summary_path = OUT_DIR / "eval_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[saved] {summary_path}")
