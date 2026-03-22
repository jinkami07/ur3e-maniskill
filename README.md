# ur3e-maniskill

UR3e ロボット × ManiSkill3 シミュレーターを使った Physical AI 研究プロジェクト。

---

## プロジェクト構成

| ディレクトリ | 手法 | 状態 |
|---|---|---|
| [`pi0/`](pi0/README.md) | **模倣学習 (pi0 LoRA)** | v4 訓練中 🔄 |
| [`tdmpc2/`](tdmpc2/README.md) | **強化学習 (TD-MPC2)** | 実装済み、未実行 🆕 |

---

## 共有リソース

```
ur3e-maniskill/
├── src/
│   └── ur3e_agent.py          # UR3e エージェント定義 ← 両手法で共有
├── scripts/
│   ├── 01_setup.sh
│   └── generate_ur3e_urdf.py  # UR3e URDF 生成 ← 両手法で共有
├── pi0/                       # 模倣学習 → pi0/README.md
│   ├── src/
│   ├── scripts/
│   └── docker/
└── tdmpc2/                    # 強化学習 → tdmpc2/README.md
    ├── src/
    ├── scripts/
    └── docker/
```

---

## ロボット仕様

| 項目 | 値 |
|---|---|
| ロボット | UR3e (6 DOF) + 平行グリッパー |
| アクション | EEF delta pose 7次元 `[dx, dy, dz, droll, dpitch, dyaw, gripper]` |
| シミュレーター | ManiSkill3 (SAPIEN) |
| GPU 環境 | GCP g2-standard-4 (L4 24GB, asia-northeast1-b) |
