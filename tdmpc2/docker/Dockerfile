# TD-MPC2 + ManiSkill3 + UR3e
# openpi/JAX は不要なので pi0 用より軽量

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential \
    libgl1-mesa-glx libegl1-mesa libgles2-mesa \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg xacro \
    && rm -rf /var/lib/apt/lists/*

# Miniforge (Python 3.11)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh \
    && bash /tmp/mf.sh -b -p /opt/conda && rm /tmp/mf.sh
ENV PATH=/opt/conda/bin:$PATH

# conda 環境 (rl)
RUN conda create -n rl python=3.11 -y
SHELL ["conda", "run", "-n", "rl", "/bin/bash", "-c"]

# PyTorch + ManiSkill3
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
    mani-skill \
    gymnasium \
    "numpy<2" \
    pyyaml \
    wandb \
    imageio[ffmpeg] \
    tqdm \
    omegaconf \
    hydra-core \
    xacro \
    yourdfpy

# TD-MPC2 公式パッケージ (存在する場合)
RUN pip install --no-cache-dir tdmpc2 || true

# UR3e URDF 生成
COPY scripts/generate_ur3e_urdf.py /tmp/generate_ur3e_urdf.py
RUN git clone -b melodic-devel --depth 1 \
    https://github.com/ros-industrial/universal_robot.git \
    /opt/universal_robot \
    && conda run -n rl python /tmp/generate_ur3e_urdf.py \
    && echo "URDF generated at /opt/ur3e/ur3e_with_gripper.urdf"

ENV MANI_SKILL_RENDER_BACKEND=egl
ENV UR3E_URDF=/opt/ur3e/ur3e_with_gripper.urdf

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
