"""
UR3e PickCube 環境ラッパー (TD-MPC2 用)

ManiSkill3 の PickCube-v1 を UR3e ロボットで使えるようにラップし、
TD-MPC2 に適した状態観測・報酬に整える。

観測 (state ベース, 34 dims):
  - agent.qpos   (8)  : アーム6 + グリッパー2
  - agent.qvel   (8)  : 関節速度
  - tcp_pose     (7)  : EEF pos(3) + quat(4)
  - cube_pos     (3)  : キューブ位置
  - cube_quat    (4)  : キューブ姿勢
  - goal_pos     (3)  : ゴール位置
  - is_grasped   (1)  : 把持フラグ

アクション (7 dims):
  pd_ee_delta_pose: [dx, dy, dz, droll, dpitch, dyaw, gripper]
"""

import gymnasium as gym
import numpy as np
import sys, os

# UR3e エージェントを登録するために import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
import ur3e_agent  # noqa: F401


class UR3ePickCubeEnv(gym.Wrapper):
    """
    PickCube-v1 + UR3e の gymnasium ラッパー。
    観測を flat numpy ベクトルに変換し、TD-MPC2 が扱いやすい形にする。
    """

    OBS_DIM = 34
    ACT_DIM = 7

    def __init__(self, env_kwargs: dict | None = None):
        kwargs = dict(
            obs_mode="state",
            robot_uids="ur3e",
            control_mode="pd_ee_delta_pose",
            render_mode=None,
            max_episode_steps=200,
        )
        if env_kwargs:
            kwargs.update(env_kwargs)

        base_env = gym.make("PickCube-v1", **kwargs)
        super().__init__(base_env)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.ACT_DIM,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def _flatten_obs(self, obs: dict) -> np.ndarray:
        agent = obs["agent"]
        extra = obs["extra"]

        qpos = np.asarray(agent["qpos"]).reshape(-1)[:8]
        qvel = np.asarray(agent["qvel"]).reshape(-1)[:8]
        tcp  = np.asarray(extra["tcp_pose"]).reshape(-1)[:7]

        cube_pos  = np.asarray(extra.get("obj_pose", extra.get("cube_pose", np.zeros(7)))).reshape(-1)[:3]
        cube_quat = np.asarray(extra.get("obj_pose", extra.get("cube_pose", np.zeros(7)))).reshape(-1)[3:7]
        goal_pos  = np.asarray(extra.get("goal_pos", np.zeros(3))).reshape(-1)[:3]
        grasped   = np.array([float(np.asarray(extra.get("is_grasped", 0)).flat[0])])

        vec = np.concatenate([qpos, qvel, tcp, cube_pos, cube_quat, goal_pos, grasped])

        # 長さを OBS_DIM に揃える（不足はゼロ埋め）
        if len(vec) < self.OBS_DIM:
            vec = np.concatenate([vec, np.zeros(self.OBS_DIM - len(vec))])
        return vec[:self.OBS_DIM].astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), float(reward), terminated, truncated, info


def make_env(seed: int = 0, **kwargs):
    """単一環境ファクトリ (gymnasium.vector 向け)"""
    def _init():
        env = UR3ePickCubeEnv(kwargs)
        env.reset(seed=seed)
        return env
    return _init
