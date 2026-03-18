FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# ── system deps ───────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        tmux \
        htop \
        unzip \
        ffmpeg \
        # EGL for headless rendering (ManiSkill3 / SAPIEN)
        libegl1 \
        libgles2 \
        libgl1 \
        libopengl0 \
        libglib2.0-0 \
        libglfw3 \
        libglew-dev \
        # Vulkan (SAPIEN preferred renderer)
        libvulkan1 \
        vulkan-tools \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── entrypoint ────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Miniforge ─────────────────────────────────────────────────────────────────
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && ${CONDA_DIR}/bin/conda clean -afy
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# ── pi0 conda env (Python 3.11) ───────────────────────────────────────────────
RUN conda create -n pi0 python=3.11 -y
SHELL ["conda", "run", "-n", "pi0", "/bin/bash", "-c"]

# ── 1. numpy pin ──────────────────────────────────────────────────────────────
RUN pip install --upgrade "numpy<2.0.0,>=1.22.4"

# ── 2. PyTorch (CUDA 12.1) ───────────────────────────────────────────────────
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ── 3. uv + openpi ───────────────────────────────────────────────────────────
RUN pip install uv
WORKDIR /opt
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
        https://github.com/Physical-Intelligence/openpi.git
WORKDIR /opt/openpi
RUN GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT=/opt/conda/envs/pi0 uv sync \
    && GIT_LFS_SKIP_SMUDGE=1 uv pip install --python /opt/conda/envs/pi0/bin/python -e .

# ── 4. re-pin (openpi may upgrade) ───────────────────────────────────────────
RUN pip install --upgrade "numpy<2.0.0,>=1.22.4"
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ── 5. ManiSkill3 + utilities ────────────────────────────────────────────────
RUN pip install \
        "mani-skill" \
        gymnasium \
        "imageio[ffmpeg]" \
        xacro \
        yourdfpy \
        pytest \
        matplotlib \
        Pillow \
        opencv-python-headless

# ── 5b. pinocchio (EEF delta 制御の IK に必要) ───────────────────────────────
RUN conda install -n pi0 -c conda-forge pinocchio -y

# ── 6. UR3e URDF (arm only, from ros-industrial/universal_robot) ──────────────
WORKDIR /opt
RUN git clone --depth 1 -b melodic-devel \
        https://github.com/ros-industrial/universal_robot.git /opt/universal_robot

# Process xacro → URDF, fix package:// paths, append simple parallel gripper
RUN mkdir -p /opt/ur3e
COPY scripts/generate_ur3e_urdf.py /opt/generate_ur3e_urdf.py
RUN python /opt/generate_ur3e_urdf.py

# ── env vars ──────────────────────────────────────────────────────────────────
ENV MANI_SKILL_RENDER_BACKEND=egl
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
ENV MPLBACKEND=Agg
ENV CONDA_DEFAULT_ENV=pi0
ENV UR3E_URDF=/opt/ur3e/ur3e_with_gripper.urdf

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
