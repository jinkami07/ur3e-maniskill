# pi0 — 模倣学習プロジェクト

PickCube-v1 タスクを **スクリプテッドポリシーでデモ収集** → **pi0 LoRA ファインチューン** → **ロールアウト評価** するパイプライン。

---

## 現在の状態 (2026-03-22)

### 完了済み ✅
- [x] UR3e エージェント定義 (`src/ur3e_agent.py`)
- [x] デモ収集スクリプト (`scripts/collect_pickcube_demos.py`)
  - Panda ロボット + PickCube-v1
  - goal_site 非表示、グリッパーロック
- [x] LeRobot 変換 (`scripts/convert_demos_to_lerobot.py`)
- [x] norm_stats 計算 (`scripts/compute_norm_stats.py`)
- [x] pi0 訓練設定 (`src/pickcube_openpi_config.py`)
- [x] 訓練パイプライン (`scripts/train_pickcube.py`)

### 進行中 🔄
- [ ] **v4 訓練** (GCP L4, 100k steps)
  - pi0_libero から LoRA ファインチューン
  - 2000デモ (38,672 フレーム)
  - wandb: `ur3e-pickcube` / run `lora_ft_v4`
  - チェックポイント: 25k / 50k / 75k / 100k
  - 完了予定: 約18〜19時間後

### 未着手 📋
- [ ] v4 ロールアウト評価 (`scripts/run_rollout_eval.py`)
- [ ] UR3e への転移（Panda → UR3e）

---

## 過去の実験

| バージョン | 内容 | 結果 |
|---|---|---|
| v1–v3 | pi0 LoRA, Panda, 500デモ | loss ~0.22 だが成功率 0/10 |
| v3 失敗原因 | グリッパー複数回開閉・green ball が画像に写る | データ品質の問題 |
| v4 (現在) | データ修正: goal_site 非表示 + グリッパーロック + 2000デモ | 訓練中 |

---

## ファイル構成

```
pi0/
├── scripts/
│   ├── collect_pickcube_demos.py   # デモ収集 (Panda + PickCube)
│   ├── convert_demos_to_lerobot.py # HDF5 → LeRobot v2 変換
│   ├── compute_norm_stats.py       # norm_stats.json 生成
│   ├── train_pickcube.py           # openpi 訓練エントリポイント
│   ├── run_rollout_eval.py         # ロールアウト評価 + wandb
│   └── save_demo_videos.py         # デモ動画確認用
├── src/
│   ├── pickcube_openpi_config.py   # TrainConfig 定義 (lora_ft_v4)
│   ├── 01_smoke_test.py
│   └── 02_ur3e_pi0_eval.py
└── docker/
    ├── Dockerfile                  # CUDA 12.1 + JAX + openpi + ManiSkill3
    └── docker-compose.yml

# 共有リソース (プロジェクトルート)
src/
└── ur3e_agent.py               # UR3e エージェント (共有)
```

---

## 訓練設定 (v4)

| パラメータ | 値 |
|---|---|
| ベースモデル | pi0_libero (pretrained) |
| Fine-tune 方式 | LoRA (gemma_2b_lora + gemma_300m_lora) |
| action_dim | 7 (EEF delta 3+3+gripper) |
| action_horizon | 8 |
| batch_size | 1 (L4 24GB) |
| steps | 100,000 |
| save_interval | 25,000 |
| データ | 2000デモ (38,672フレーム) |
| プロンプト | "pick up the red cube" |

---

## 次のステップ

1. v4 訓練完了後 → ロールアウト評価で成功率を確認
2. 成功率が高ければ UR3e への転移を検討
3. 低ければ → データ追加 or **RFT (Reinforcement Fine-Tuning)** へ

---

## 関連リンク

- [openpi GitHub](https://github.com/Physical-Intelligence/openpi)
- [WandB プロジェクト](https://wandb.ai/tsodw-totwis07-the-university-of-tokyo/ur3e-pickcube)
