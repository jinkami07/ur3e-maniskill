"""
TD-MPC2 training script for UR3e PickCube (ManiSkill3)

Usage (inside container):
  python tdmpc2/scripts/train_tdmpc2.py

環境変数:
  WANDB_API_KEY     W&B キー
  CHECKPOINT_DIR    チェックポイント保存先 (default: /opt/checkpoints/tdmpc2)
  NUM_ENVS          並列環境数 (default: 8)
  STEPS             総ステップ数 (default: 1_000_000)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import collections

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))          # ur3e_agent, ur3e_pickcube_env
sys.path.insert(0, str(ROOT / "tdmpc2/src"))

from ur3e_pickcube_env import make_env

# ── Config ───────────────────────────────────────────────────────────────────

CFG_PATH = ROOT / "tdmpc2/cfgs/config.yaml"
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

# 環境変数オーバーライド
STEPS         = int(os.environ.get("STEPS",        cfg["steps"]))
NUM_ENVS      = int(os.environ.get("NUM_ENVS",     cfg["num_envs"]))
CKPT_DIR      = Path(os.environ.get("CHECKPOINT_DIR", cfg["checkpoint_dir"]))
EVAL_FREQ     = int(cfg["eval_freq"])
EVAL_EPS      = int(cfg["eval_episodes"])
SAVE_FREQ     = int(cfg["save_freq"])
SEED_STEPS    = int(cfg["seed_steps"])
OBS_DIM       = int(cfg["obs_dim"])
ACT_DIM       = int(cfg["action_dim"])
BATCH_SIZE    = int(cfg["batch_size"])
HORIZON       = int(cfg["horizon"])
LR            = float(cfg["lr"])
DISCOUNT      = float(cfg["discount"])
TAU           = float(cfg["tau"])
GRAD_CLIP     = float(cfg["grad_clip_norm"])

CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[tdmpc2] Device: {DEVICE}, Envs: {NUM_ENVS}, Steps: {STEPS}")

# ── W&B ──────────────────────────────────────────────────────────────────────

import wandb
wandb.init(
    project=cfg["wandb_project"],
    name=cfg["exp_name"],
    config=cfg,
    mode="online" if cfg.get("wandb") else "disabled",
)

# ── TD-MPC2 モデル ─────────────────────────────────────────────────────────────
# pip install tdmpc2 (Hansen et al. 公式パッケージ)

try:
    from tdmpc2 import TDMPC2
    from tdmpc2.common.buffer import Buffer
    from omegaconf import OmegaConf
    _USE_OFFICIAL = True
    print("[tdmpc2] Using official tdmpc2 package")
except ImportError:
    _USE_OFFICIAL = False
    print("[tdmpc2] Official tdmpc2 not found. Using minimal fallback implementation.")


# ── Minimal fallback (SAC-like) if tdmpc2 not installed ──────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act  = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew  = np.zeros((capacity, 1),       dtype=np.float32)
        self.next = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1),       dtype=np.float32)
        self.ptr  = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        n = len(obs)
        idx = np.arange(self.ptr, self.ptr + n) % self.capacity
        self.obs[idx]  = obs
        self.act[idx]  = act
        self.rew[idx]  = np.array(rew).reshape(-1, 1)
        self.next[idx] = next_obs
        self.done[idx] = np.array(done, dtype=float).reshape(-1, 1)
        self.ptr  = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(DEVICE),
            torch.FloatTensor(self.act[idx]).to(DEVICE),
            torch.FloatTensor(self.rew[idx]).to(DEVICE),
            torch.FloatTensor(self.next[idx]).to(DEVICE),
            torch.FloatTensor(self.done[idx]).to(DEVICE),
        )


def _mlp(in_dim, out_dim, hidden=512, layers=3):
    dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
    mods = []
    for i in range(len(dims) - 1):
        mods.append(torch.nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            mods.append(torch.nn.LayerNorm(dims[i+1]))
            mods.append(torch.nn.Mish())
    return torch.nn.Sequential(*mods)


class SimpleTDMPC2(torch.nn.Module):
    """TD-MPC2 の簡易実装 (公式パッケージがない場合のフォールバック)。
    世界モデル + ポリシー + Q値をまとめて学習する。"""

    def __init__(self, obs_dim, act_dim, latent=512, hidden=512):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # エンコーダ: obs → z
        self.encoder = _mlp(obs_dim, latent, hidden)
        # 世界モデル: z + a → z'
        self.dynamics = _mlp(latent + act_dim, latent, hidden)
        # 報酬ヘッド
        self.reward_head = _mlp(latent + act_dim, 1, hidden)
        # ポリシー (tanh squash)
        self.policy = torch.nn.Sequential(
            _mlp(latent, act_dim * 2, hidden),
        )
        # Q値 (x2 でクリッピング)
        self.q1 = _mlp(latent + act_dim, 1, hidden)
        self.q2 = _mlp(latent + act_dim, 1, hidden)

        # ターゲットネットワーク
        import copy
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.enc_target = copy.deepcopy(self.encoder)
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()) + list(self.enc_target.parameters()):
            p.requires_grad_(False)

    def encode(self, obs):
        return self.encoder(obs)

    def next_z(self, z, a):
        return self.dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z, a):
        return self.reward_head(torch.cat([z, a], dim=-1))

    def pi(self, z):
        out = self.policy(z)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        a = dist.rsample()
        log_prob = dist.log_prob(a).sum(-1, keepdim=True)
        a = torch.tanh(a)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        return a, log_prob

    def q(self, z, a):
        za = torch.cat([z, a], dim=-1)
        return self.q1(za), self.q2(za)

    def q_target(self, z, a):
        za = torch.cat([z, a], dim=-1)
        return self.q1_target(za), self.q2_target(za)

    def soft_update(self, tau=0.01):
        for src, tgt in [(self.q1, self.q1_target), (self.q2, self.q2_target), (self.encoder, self.enc_target)]:
            for ps, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.copy_(tau * ps.data + (1 - tau) * pt.data)


# ── 環境 ──────────────────────────────────────────────────────────────────────

import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs  # noqa: task 登録

print("[tdmpc2] Creating vectorized environments...")
vec_env = gym.make_vec(
    "PickCube-v1",
    num_envs=NUM_ENVS,
    obs_mode="state",
    robot_uids="ur3e",
    control_mode="pd_ee_delta_pose",
    max_episode_steps=cfg["episode_length"],
    vectorization_mode="async",
)
eval_env = gym.make(
    "PickCube-v1",
    obs_mode="state",
    robot_uids="ur3e",
    control_mode="pd_ee_delta_pose",
    max_episode_steps=cfg["episode_length"],
    render_mode=None,
)
print("[tdmpc2] Environments ready.")

# ── エージェント初期化 ─────────────────────────────────────────────────────────

if _USE_OFFICIAL:
    _cfg = OmegaConf.create({
        "obs": "state", "obs_dim": OBS_DIM, "action_dim": ACT_DIM,
        "latent_dim": cfg["latent_dim"], "mlp_dim": cfg["mlp_dim"],
        "num_enc_layers": cfg["num_enc_layers"], "num_bins": cfg["num_bins"],
        "discount": DISCOUNT, "lr": LR, "tau": TAU,
        "num_pi_trajs": cfg["num_pi_trajs"],
        "num_samples": cfg["num_samples"],
        "num_iterations": cfg["num_iterations"],
        "temperature": cfg["temperature"],
        "momentum": cfg["momentum"],
        "min_std": cfg["min_std"], "max_std": cfg["max_std"],
        "horizon": HORIZON, "rho": cfg["rho"],
        "grad_clip_norm": GRAD_CLIP,
        "device": DEVICE,
        "episode_length": cfg["episode_length"],
    })
    agent = TDMPC2(_cfg).to(DEVICE)
    buffer = Buffer(_cfg)
else:
    agent = SimpleTDMPC2(OBS_DIM, ACT_DIM).to(DEVICE)
    buffer = ReplayBuffer(cfg["buffer_size"], OBS_DIM, ACT_DIM)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    alpha_opt = torch.optim.Adam([log_alpha], lr=LR)
    target_entropy = -ACT_DIM * 0.5


# ── ヘルパー: 観測をフラット化 ────────────────────────────────────────────────

def _flatten(obs):
    if isinstance(obs, dict):
        agent_obs = obs.get("agent", {})
        extra     = obs.get("extra", {})
        parts = []
        for k in ["qpos", "qvel"]:
            v = agent_obs.get(k)
            if v is not None:
                parts.append(np.asarray(v).reshape(NUM_ENVS, -1))
        for k in ["tcp_pose", "obj_pose", "goal_pos", "is_grasped"]:
            v = extra.get(k)
            if v is not None:
                parts.append(np.asarray(v).reshape(NUM_ENVS, -1))
        flat = np.concatenate(parts, axis=-1).astype(np.float32)
        # OBS_DIM に揃える
        if flat.shape[-1] < OBS_DIM:
            pad = np.zeros((NUM_ENVS, OBS_DIM - flat.shape[-1]), dtype=np.float32)
            flat = np.concatenate([flat, pad], axis=-1)
        return flat[:, :OBS_DIM]
    return np.asarray(obs, dtype=np.float32).reshape(NUM_ENVS, -1)[:, :OBS_DIM]

def _flatten_single(obs):
    if isinstance(obs, dict):
        agent_obs = obs.get("agent", {})
        extra     = obs.get("extra", {})
        parts = []
        for k in ["qpos", "qvel"]:
            v = agent_obs.get(k)
            if v is not None:
                parts.append(np.asarray(v).reshape(-1))
        for k in ["tcp_pose", "obj_pose", "goal_pos", "is_grasped"]:
            v = extra.get(k)
            if v is not None:
                parts.append(np.asarray(v).reshape(-1))
        flat = np.concatenate(parts).astype(np.float32)
        if len(flat) < OBS_DIM:
            flat = np.concatenate([flat, np.zeros(OBS_DIM - len(flat))])
        return flat[:OBS_DIM]
    return np.asarray(obs, dtype=np.float32).reshape(-1)[:OBS_DIM]


# ── 評価ループ ────────────────────────────────────────────────────────────────

def evaluate(step: int) -> float:
    agent.eval()
    successes = 0
    ep_rewards_list = []
    ep_lengths = []
    ep_is_grasped = []
    ep_tcp_cube_dists = []

    for ep in range(EVAL_EPS):
        obs, _ = eval_env.reset(seed=9000 + ep)
        done = False
        t = 0
        ep_reward = 0.0
        while not done and t < cfg["episode_length"]:
            flat_obs = _flatten_single(obs)
            o = torch.FloatTensor(flat_obs).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                if _USE_OFFICIAL:
                    a = agent.act(o, t0=(t == 0), eval_mode=True).cpu().numpy().reshape(-1)
                else:
                    z = agent.encode(o)
                    a, _ = agent.pi(z)
                    a = a.cpu().numpy().reshape(-1)
            obs, reward, terminated, truncated, info = eval_env.step(a)
            ep_reward += float(reward)
            done = terminated or truncated
            t += 1

        # obs layout: qpos(8), qvel(8), tcp(7), cube_pos(3), cube_quat(4), goal_pos(3), grasped(1)
        flat_final = _flatten_single(obs)
        tcp_pos  = flat_final[16:19]
        cube_pos = flat_final[23:26]
        tcp_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))

        successes += float(info.get("success", False))
        ep_rewards_list.append(ep_reward)
        ep_lengths.append(t)
        ep_is_grasped.append(float(info.get("is_grasped", False)))
        ep_tcp_cube_dists.append(tcp_cube_dist)

    rate = successes / EVAL_EPS
    mean_reward = float(np.mean(ep_rewards_list))
    mean_length = float(np.mean(ep_lengths))
    mean_is_grasped = float(np.mean(ep_is_grasped))
    mean_tcp_cube_dist = float(np.mean(ep_tcp_cube_dists))

    print(f"[eval] step={step:>7d}  success_rate={rate:.2f}  ({int(successes)}/{EVAL_EPS})"
          f"  mean_reward={mean_reward:.3f}  mean_dist={mean_tcp_cube_dist:.4f}")

    log = {
        "eval/success_rate":       rate,
        "eval/mean_reward":        mean_reward,
        "eval/mean_episode_length": mean_length,
        "eval/mean_is_grasped":    mean_is_grasped,
        "eval/mean_tcp_cube_dist": mean_tcp_cube_dist,
        "eval/num_successes":      int(successes),
    }
    # Per-episode detail (first 5 episodes)
    for ep_i, (r, l, g, d) in enumerate(zip(
            ep_rewards_list[:5], ep_lengths[:5], ep_is_grasped[:5], ep_tcp_cube_dists[:5])):
        log[f"eval/ep{ep_i}_reward"]        = r
        log[f"eval/ep{ep_i}_length"]        = l
        log[f"eval/ep{ep_i}_is_grasped"]    = g
        log[f"eval/ep{ep_i}_tcp_cube_dist"] = d
    wandb.log(log, step=step)
    agent.train()
    return rate


# ── メイン学習ループ ───────────────────────────────────────────────────────────

print("[tdmpc2] Starting training...")
obs_arr, _ = vec_env.reset(seed=0)
obs_arr = _flatten(obs_arr)
ep_rewards = np.zeros(NUM_ENVS)
global_step = 0
num_episodes = 0
reward_history = collections.deque(maxlen=100)  # rolling mean over last 100 episodes
t0 = time.time()

while global_step < STEPS:
    # ── アクション選択 ────────────────────────────────────────────────────────
    if global_step < SEED_STEPS:
        actions = np.random.uniform(-1, 1, (NUM_ENVS, ACT_DIM)).astype(np.float32)
    else:
        agent.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_arr).to(DEVICE)
            if _USE_OFFICIAL:
                actions = agent.act(obs_t, t0=True, eval_mode=False).cpu().numpy()
            else:
                z = agent.encode(obs_t)
                actions, _ = agent.pi(z)
                actions = actions.cpu().numpy()
        agent.train()

    # ── 環境ステップ ─────────────────────────────────────────────────────────
    next_obs_raw, rewards, terms, truncs, infos = vec_env.step(actions)
    next_obs_arr = _flatten(next_obs_raw)
    dones = terms | truncs

    buffer.add(obs_arr, actions, rewards, next_obs_arr, dones)

    ep_rewards += rewards
    obs_arr = next_obs_arr
    global_step += NUM_ENVS

    # リセット (done 環境)
    if np.any(dones):
        done_idx = np.where(dones)[0]
        for idx in done_idx:
            reward_history.append(float(ep_rewards[idx]))
            num_episodes += 1
        mean_r = ep_rewards[done_idx].mean()
        mean_r_100 = float(np.mean(reward_history)) if reward_history else 0.0
        wandb.log({
            "train/episode_reward":    float(mean_r),
            "train/mean_reward_100ep": mean_r_100,
            "train/num_episodes":      num_episodes,
            "train/step":              global_step,
        }, step=global_step)
        ep_rewards[done_idx] = 0

    # ── 学習ステップ ──────────────────────────────────────────────────────────
    if global_step >= SEED_STEPS and buffer.size >= BATCH_SIZE:
        if _USE_OFFICIAL:
            metrics = agent.update(buffer)
            if global_step % 1000 == 0:
                train_log = {f"train/{k}": v for k, v in metrics.items()}
                train_log["train/buffer_size"] = buffer.size
                train_log["train/sps"]         = global_step / (time.time() - t0)
                wandb.log(train_log, step=global_step)
        else:
            # シンプル SAC 風更新
            obs_b, act_b, rew_b, next_b, done_b = buffer.sample(BATCH_SIZE)
            with torch.no_grad():
                z_next = agent.enc_target(next_b)
                a_next, logp_next = agent.pi(z_next)
                alpha = log_alpha.exp()
                q1_t, q2_t = agent.q_target(z_next, a_next)
                q_target = rew_b + DISCOUNT * (1 - done_b) * (torch.min(q1_t, q2_t) - alpha * logp_next)

            z_b = agent.encode(obs_b)
            q1_val, q2_val = agent.q(z_b.detach(), act_b)
            q_loss = torch.nn.functional.mse_loss(q1_val, q_target) + torch.nn.functional.mse_loss(q2_val, q_target)

            a_pi, logp_pi = agent.pi(z_b.detach())
            q1_pi, q2_pi = agent.q(z_b.detach(), a_pi)
            pi_loss = (log_alpha.exp().detach() * logp_pi - torch.min(q1_pi, q2_pi)).mean()
            alpha_loss = -(log_alpha * (logp_pi + target_entropy).detach()).mean()

            loss = q_loss + pi_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)
            optimizer.step()

            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

            agent.soft_update(TAU)

            if global_step % 1000 == 0:
                wandb.log({
                    "train/q_loss":      q_loss.item(),
                    "train/pi_loss":     pi_loss.item(),
                    "train/alpha":       log_alpha.exp().item(),
                    "train/alpha_loss":  alpha_loss.item(),
                    "train/buffer_size": buffer.size,
                    "train/sps":         global_step / (time.time() - t0),
                }, step=global_step)

    # ── 評価 ────────────────────────────────────────────────────────────────
    if global_step % EVAL_FREQ < NUM_ENVS:
        evaluate(global_step)

    # ── チェックポイント ─────────────────────────────────────────────────────
    if global_step % SAVE_FREQ < NUM_ENVS:
        ckpt = CKPT_DIR / f"step_{global_step:08d}.pt"
        torch.save({"step": global_step, "model": agent.state_dict()}, ckpt)
        print(f"[tdmpc2] Saved checkpoint: {ckpt}")

    # ── 定期ログ (テキスト + wandb) ────────────────────────────────────────────
    if global_step % 10_000 < NUM_ENVS:
        elapsed = time.time() - t0
        sps = global_step / elapsed
        mean_r_100 = float(np.mean(reward_history)) if reward_history else 0.0
        print(f"[tdmpc2] step={global_step:>7d}  sps={sps:.0f}  buffer={buffer.size}"
              f"  episodes={num_episodes}  mean_r100={mean_r_100:.3f}")
        wandb.log({
            "train/sps":               sps,
            "train/buffer_size":       buffer.size,
            "train/num_episodes":      num_episodes,
            "train/mean_reward_100ep": mean_r_100,
        }, step=global_step)

vec_env.close()
eval_env.close()
print("[tdmpc2] Training complete.")
wandb.finish()
