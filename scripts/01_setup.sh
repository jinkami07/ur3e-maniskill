#!/bin/bash
# ---------------------------------------------------------------------------
# 01_setup.sh
#
# Dockerコンテナ外でセットアップする場合のスクリプト。
# 通常はDockerfileで自動実行されるため不要。
# GCPホスト上で直接 conda env を使う場合に使用。
#
# 前提: conda env "pi0" 作成済み, openpi インストール済み
# ---------------------------------------------------------------------------
set -euo pipefail

echo "============================================================"
echo " Installing ManiSkill3 into conda env: pi0"
echo "============================================================"

conda run --no-capture-output -n pi0 pip install "mani-skill" xacro yourdfpy

echo ""
echo "============================================================"
echo " Downloading UR3e URDF (ros-industrial/universal_robot)"
echo "============================================================"

if [ ! -d "/opt/universal_robot" ]; then
    git clone --depth 1 -b noetic-devel \
        https://github.com/ros-industrial/universal_robot.git \
        /opt/universal_robot
    echo "[OK] Cloned universal_robot"
else
    echo "[skip] /opt/universal_robot already exists"
fi

echo ""
echo "============================================================"
echo " Generating UR3e URDF with gripper"
echo "============================================================"

conda run --no-capture-output -n pi0 python scripts/generate_ur3e_urdf.py

echo ""
echo "============================================================"
echo " Smoke test: ManiSkill3 import"
echo "============================================================"

conda run --no-capture-output -n pi0 python -c "
import mani_skill
import gymnasium as gym
import mani_skill.envs
print('mani_skill version:', mani_skill.__version__)
print('[OK] ManiSkill3 import OK')
"

echo ""
echo "============================================================"
echo " Setup complete. Run:"
echo "   python src/01_smoke_test.py"
echo "============================================================"
