#!/usr/bin/env bash
# =============================================================================
# PickCube-v1 フルトレーニングパイプライン
#
# 実行順:
#   1. デモ収集 (Panda oracle, 500 episodes) → /opt/pickcube_demos/demos.h5
#   2. LeRobot 形式に変換 → /opt/pickcube_lerobot
#   3. pi0 LoRA 訓練 (wandb 有効)
#   4. VM シャットダウン (訓練終了後)
#
# 使い方 (GCP instance 上で):
#   bash scripts/run_training_pipeline.sh
#
# 環境変数:
#   WANDB_API_KEY   - wandb API キー (必須)
#   NUM_DEMOS       - 収集するデモ数 (default: 500)
#   SKIP_COLLECT    - "1" でデモ収集をスキップ
#   SKIP_CONVERT    - "1" で LeRobot 変換をスキップ
# =============================================================================
set -euo pipefail

# ── 設定 ──────────────────────────────────────────────────────────────────────
NUM_DEMOS="${NUM_DEMOS:-500}"
DEMOS_H5="/opt/pickcube_demos/demos.h5"
LEROBOT_DIR="/opt/pickcube_lerobot"
CKPT_DIR="/opt/checkpoints"
LOG_DIR="/tmp/pipeline_logs"
WANDB_API_KEY="${WANDB_API_KEY:-}"

# Docker image (build が済んでいる前提)
DOCKER_IMAGE="docker-sandbox:latest"

log() { echo "[pipeline] $(date '+%H:%M:%S') $*"; }

mkdir -p "$LOG_DIR"

# ── wandb API キー読み込み ────────────────────────────────────────────────────
if [ -z "$WANDB_API_KEY" ] && [ -f /opt/wandb_api_key.txt ]; then
    WANDB_API_KEY="$(cat /opt/wandb_api_key.txt)"
    log "Loaded WANDB_API_KEY from /opt/wandb_api_key.txt"
fi

if [ -z "$WANDB_API_KEY" ]; then
    log "WARNING: WANDB_API_KEY not set. wandb logging will be offline."
fi

# ── Step 1: デモ収集 ──────────────────────────────────────────────────────────
if [ "${SKIP_COLLECT:-0}" = "1" ] && [ -f "$DEMOS_H5" ]; then
    log "Step 1: SKIP (demos already exist at $DEMOS_H5)"
else
    log "Step 1: Collecting $NUM_DEMOS demos → $DEMOS_H5"
    mkdir -p "$(dirname "$DEMOS_H5")"

    docker run --rm \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e MANI_SKILL_RENDER_BACKEND=egl \
        -v /opt/pickcube_demos:/opt/pickcube_demos \
        -v "$(pwd)/scripts:/workspace/scripts" \
        -v "$(pwd)/src:/workspace/src" \
        -w /workspace \
        "$DOCKER_IMAGE" \
        python scripts/collect_pickcube_demos.py \
            --num-demos "$NUM_DEMOS" \
            --out "$DEMOS_H5" \
        2>&1 | tee "$LOG_DIR/collect.log"

    log "Step 1: Done"
fi

# ── Step 2: LeRobot 変換 ──────────────────────────────────────────────────────
if [ "${SKIP_CONVERT:-0}" = "1" ] && [ -d "$LEROBOT_DIR" ]; then
    log "Step 2: SKIP (dataset already exists at $LEROBOT_DIR)"
else
    log "Step 2: Converting HDF5 → LeRobot format → $LEROBOT_DIR"
    mkdir -p "$LEROBOT_DIR"

    docker run --rm \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -v /opt/pickcube_demos:/opt/pickcube_demos \
        -v /opt/pickcube_lerobot:/opt/pickcube_lerobot \
        -v "$(pwd)/scripts:/workspace/scripts" \
        -w /workspace \
        "$DOCKER_IMAGE" \
        python scripts/convert_demos_to_lerobot.py \
            --demos "$DEMOS_H5" \
            --out "$LEROBOT_DIR" \
        2>&1 | tee "$LOG_DIR/convert.log"

    log "Step 2: Done"
fi

# ── Step 3: 訓練 ──────────────────────────────────────────────────────────────
log "Step 3: Starting pi0 LoRA training (wandb enabled)"

docker run --rm \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e MANI_SKILL_RENDER_BACKEND=egl \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e WANDB_MODE=offline \
    -e HF_LEROBOT_HOME=/opt \
    -v /opt/pickcube_lerobot:/opt/pickcube_lerobot \
    -v /opt/checkpoints:/opt/checkpoints \
    -v "$(pwd)/scripts:/workspace/scripts" \
    -v "$(pwd)/src:/workspace/src" \
    -v /tmp/train_output:/workspace/output \
    -w /workspace \
    --shm-size=4g \
    "$DOCKER_IMAGE" \
    python scripts/train_pickcube.py \
    2>&1 | tee "$LOG_DIR/train.log"

TRAIN_EXIT="${PIPESTATUS[0]}"

if [ "$TRAIN_EXIT" -eq 0 ]; then
    log "Step 3: Training completed successfully"
else
    log "Step 3: Training FAILED (exit code $TRAIN_EXIT)"
fi

# ── Step 4: VM シャットダウン ─────────────────────────────────────────────────
log "Pipeline finished. Shutting down VM in 60 seconds ..."
log "(Ctrl-C to cancel shutdown)"
sleep 60
sudo poweroff
