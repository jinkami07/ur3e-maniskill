# TD-MPC2 — 強化学習プロジェクト

ManiSkill3 上の PickCube-v1 タスクを **強化学習 (TD-MPC2)** で解く。
デモ不要・報酬関数だけで学習できるのが pi0 との最大の違い。

---

## 現在の状態 (2026-03-22)

### 完了済み ✅
- [x] 環境ラッパー (`tdmpc2/src/ur3e_pickcube_env.py`)
  - PickCube-v1 + UR3e + state obs (34次元)
- [x] 訓練スクリプト (`tdmpc2/scripts/train_tdmpc2.py`)
  - 公式 `tdmpc2` パッケージ対応 + フォールバック実装
- [x] 設定ファイル (`tdmpc2/cfgs/config.yaml`)
- [x] Docker 環境 (`tdmpc2/docker/Dockerfile`, `docker-compose.yml`)
- [x] 評価スクリプト (`tdmpc2/scripts/eval_tdmpc2.py`)

### 未着手 📋
- [ ] GCP VM でビルド・実行
- [ ] 学習曲線の確認 (wandb)
- [ ] 評価 → 成功率 ≥ 80% を目標
- [ ] pi0 との性能比較

---

## TD-MPC2 とは

**TD-MPC2 (Temporal Difference Model Predictive Control v2)** は、世界モデルベースの強化学習手法。

```
[観測 o] → [エンコーダ z = f(o)]
                ↓
[世界モデル]  z' = T(z, a)   ← 次の潜在状態を予測
[報酬モデル]  r  = R(z, a)   ← 報酬を予測
[Q値]        Q  = V(z, a)   ← 価値を予測
                ↓
[MPPI プランニング] → 最良のアクション列を選択
```

- **デモ不要** (純粋 RL)
- **サンプル効率が高い** (モデルベース)
- **ManiSkill3 公式ベースライン** に採用

---

## 設計

### 観測空間 (34次元, state ベース)

| 要素 | 次元 | 内容 |
|---|---|---|
| qpos | 8 | アーム6 + グリッパー2 |
| qvel | 8 | 関節速度 |
| tcp_pose | 7 | EEF pos(3) + quat(4) |
| cube_pos | 3 | キューブ位置 |
| cube_quat | 4 | キューブ姿勢 |
| goal_pos | 3 | ゴール位置 |
| is_grasped | 1 | 把持フラグ |

### アクション空間 (7次元)
`[dx, dy, dz, droll, dpitch, dyaw, gripper]` ← pi0 と同じ

### 報酬
ManiSkill3 の PickCube-v1 組み込み dense reward を使用
（距離報酬 + 把持ボーナス + 成功報酬）

---

## 実行方法

```bash
# Docker ビルド (プロジェクトルートから)
cd tdmpc2/docker
docker compose build

# 訓練開始
WANDB_API_KEY=xxx docker compose up

# 評価
docker compose run tdmpc2 \
  python tdmpc2/scripts/eval_tdmpc2.py \
  --checkpoint /opt/checkpoints/tdmpc2/step_00100000.pt \
  --num-episodes 20 --save-video
```

---

## ハイパーパラメータ (config.yaml)

| パラメータ | 値 | 備考 |
|---|---|---|
| steps | 1,000,000 | 総ステップ |
| num_envs | 8 | 並列環境 |
| horizon | 3 | MPPI ホライゾン |
| batch_size | 256 | |
| latent_dim | 512 | 潜在空間次元 |
| discount | 0.99 | 割引率 |
| eval_freq | 10,000 | 評価間隔 |

---

## ファイル構成

```
tdmpc2/
├── README.md              ← このファイル
├── cfgs/
│   └── config.yaml        # ハイパーパラメータ
├── scripts/
│   ├── train_tdmpc2.py    # 訓練スクリプト (メイン)
│   └── eval_tdmpc2.py     # 評価スクリプト
├── src/
│   └── ur3e_pickcube_env.py  # 環境ラッパー
├── docker/
│   ├── Dockerfile         # ManiSkill3 + PyTorch (JAX 不要)
│   └── docker-compose.yml
└── output/                # 評価動画・ログ (gitignore)
```

---

## 今後の展望

```
TD-MPC2 (シミュ)
    ↓ 成功率 ≥ 80%
Sim-to-Real 転移 (UR3e 実機)
    or
RFT: pi0 (IL) + TD-MPC2 (RL) の組み合わせ
```

---

## 参考

- [TD-MPC2 論文](https://arxiv.org/abs/2310.16828)
- [TD-MPC2 GitHub](https://github.com/nicklashansen/tdmpc2)
- [ManiSkill3 TD-MPC2 ベースライン](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/tdmpc2)
