FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    LIBERO_CONFIG_PATH=/workspace/.libero \
    PYTHONPATH=/workspace/LIBERO

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    build-essential \
    cmake \
    git \
    tmux \
    ffmpeg \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglfw3 \
    libglew2.2 \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3 /usr/local/bin/python

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r requirements.txt

COPY LIBERO ./LIBERO
RUN python -m pip install -e /workspace/LIBERO

CMD ["bash"]
