# Base image: NVIDIA CUDA 12.2.2 with cuDNN 8 runtime on Ubuntu 22.04
FROM nvcr.io/nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    g++ \
    gdb \
    make \
    wget \
    curl \
    git \
    vim \
    python3 \
    python3-pip \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libjemalloc2 \
    pkg-config \
    unzip \
    ca-certificates \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* # Clean up apt cache

# Set working directory
WORKDIR /app

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI

# Change working directory to ComfyUI
WORKDIR /app/ComfyUI

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools && \
    pip3 install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu121 \
    --pre torch torchvision torchaudio && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir \
    opencv-python \
    piexif \
    numba \
    evalidate \
    accelerate \
    matplotlib \
    imageio-ffmpeg \
    gguf \
    GitPython \
    opencv-python-headless

# Set environment variables
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    PYTHONPATH="${PYTHONPATH}:/app/ComfyUI" \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX" \
    NVIDIA_VISIBLE_DEVICES=all

# Set execute permissions on the ComfyUI directory
RUN chmod -R 755 /app/ComfyUI

# Expose port 8188
EXPOSE 8188

# Run ComfyUI
CMD ["python3", "main.py", \
    "--cuda-malloc", \
    "--use-pytorch-cross-attention", \
    "--listen", "0.0.0.0", \
    "--port", "8188"]
