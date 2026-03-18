# ur3e-maniskill 作戦計画

## 目的

ManiSkill3 シミュレーション上で UR3e ロボットを動かし、
VLA モデル（pi0）の評価パイプラインを構築する。
最終的に実機 UR3e での動作も目指す。

---

## 技術スタック

| レイヤー | 採用技術 | 理由 |
|---|---|---|
| シミュレーター | **ManiSkill3** | GPU並列・軽量・URDF 対応 |
| ロボット | **UR3e** | 実機と合わせる |
| VLA モデル | **pi0** (openpi) | 既存スタックをそのまま流用 |
| アクション制御 | **EEF delta** | pi0 の出力形式と一致 |
| インフラ | **Docker + GCP** | GPU 環境 |

---

## ディレクトリ構成

```
ur3e-maniskill/
├── PLAN.md
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml       # GCP GPU 環境用
│   └── entrypoint.sh
├── scripts/
│   ├── 01_setup.sh              # Docker外でセットアップする場合
│   └── generate_ur3e_urdf.py    # UR3e URDF 生成（Dockerfile内で実行）
├── src/
│   ├── ur3e_agent.py            # ManiSkill3 用 UR3e エージェント定義
│   ├── 01_smoke_test.py         # ManiSkill3 + UR3e 動作確認
│   └── 02_ur3e_pi0_eval.py      # pi0 で UR3e を評価
└── output/                      # 動画・ログ（gitignore済み）
```

---

## 実装フェーズ

### Phase 1: 環境構築
- [ ] Dockerfile（CUDA + ManiSkill3 + openpi）
- [ ] `01_setup.sh` — ManiSkill3 + UR3e URDF セットアップ
- [ ] `01_smoke_test.py` — UR3e 環境の起動・観測確認

### Phase 2: eval パイプライン
- [ ] `02_ur3e_pi0_eval.py`
  - ManiSkill3 UR3e + PickCube タスク
  - 224×224 カメラ（front + wrist）
  - pi0 アクション → ManiSkill3 EEF デルタ制御
  - 成功率自動計測 + 動画保存

### Phase 3: 実機移行（シミュ成功後）
- [ ] 実機 UR3e との接続確認
- [ ] sim-to-real ギャップの調整

---

## eval 設計方針

```
✅ 解像度: 224×224（学習時と一致）
✅ アクション加工なし（pi0 出力そのまま）
✅ 6DoF フル制御
✅ ファインチューニング済みチェックポイント指定可
```

---

## 環境変数

| 変数 | デフォルト | 説明 |
|---|---|---|
| `CKPT_DIR` | （空） | チェックポイントのパス |
| `NUM_EPISODES` | `5` | エピソード数 |
| `TOTAL_STEPS` | `200` | 1エピソードの最大ステップ数 |
| `HORIZON` | `10` | 1回の推論で実行するアクション数 |

---

## 参考リンク

- [ManiSkill3 ドキュメント](https://maniskill.readthedocs.io/)
- [ManiSkill3 GitHub](https://github.com/haosulab/ManiSkill)
- [openpi GitHub](https://github.com/Physical-Intelligence/openpi)
- [UR3e URDF (ROS2)](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)
