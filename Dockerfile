# ComfyUI Docker Image
# ====================
#
# Multi-stage build that produces a CUDA-accelerated container for ComfyUI,
# a modular Stable Diffusion GUI and backend.
#
# Builder stage : NVIDIA CUDA 12.8.0 cuDNN -devel- on Ubuntu 22.04
#                 Compiles native extensions (xformers, bitsandbytes, etc.)
#                 against nvcc and full -dev headers, then installs all
#                 Python dependencies into /usr/local/lib/python3.12.
# Runtime stage : NVIDIA CUDA 12.8.0 cuDNN -runtime- on Ubuntu 22.04
#                 Carries only runtime shared libraries plus Python 3.12,
#                 and inherits the prebuilt site-packages and ComfyUI tree
#                 from the builder. Trims roughly 3 GB versus a single-stage
#                 -devel- image.
#
# Build:
#   docker build -t ghcr.io/memenow/comfyui-helm:v2.0.0 .
#
# Run:
#   docker run -d --gpus all -p 8188:8188 ghcr.io/memenow/comfyui-helm:v2.0.0

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# CUDA architectures used when xformers / custom kernels compile from source.
# Covers Volta (7.0) through Hopper (9.0) plus Blackwell (10.0 datacenter,
# 12.0 consumer). +PTX keeps forward compatibility with newer GPUs.
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"

# Build-time system dependencies: Python 3.12 from deadsnakes plus headers
# and toolchain required to compile native Python wheels.
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        git \
        wget \
        curl \
        unzip \
        build-essential \
        cmake \
        pkg-config \
        libssl-dev \
        libffi-dev \
        libjpeg-dev \
        libpng-dev \
        libwebp-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libavutil-dev \
        libopencv-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

WORKDIR /app

# Bootstrap pip for Python 3.12 (deadsnakes does not ship pip).
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8 wheels.
RUN python3 -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.1 \
        torchvision==0.24.1 \
        torchaudio==2.9.1

# Lock PyTorch versions globally so transitive dependencies cannot upgrade them.
RUN printf 'torch==2.9.1\ntorchvision==0.24.1\ntorchaudio==2.9.1\n' > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# Pin ComfyUI core to a specific commit for reproducible builds.
# Override with --build-arg COMFYUI_REF=<sha|tag|branch> when rebuilding.
ARG COMFYUI_REF=df22bcd5e192ce0b1ae09eaf2e423d0a12cf6638
RUN git clone https://github.com/Comfy-Org/ComfyUI.git && \
    cd /app/ComfyUI && \
    git checkout "${COMFYUI_REF}"

WORKDIR /app/ComfyUI

# ComfyUI's own Python requirements.
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install curated custom nodes during image build so the container does not
# need to clone repositories or install Python packages at startup.
# Each reference may be a branch, tag, or commit SHA for reproducible pinning.
ARG COMFYUI_MANAGER_REF=491f847bbc286588175695ea43fa4e13cd14a437
ARG COMFYUI_GGUF_REF=6ea2651e7df66d7585f6ffee804b20e92fb38b8a
ARG COMFYUI_KJNODES_REF=b7646ad70a7daa7aeb919ca542274758d26ba2df
ARG COMFYUI_LTXVIDEO_REF=2acf7af8991f33b5cc06ec26753cb6e88e057d04
ARG COMFYUI_SEEDVR2_REF=4490bd1f482e026674543386bb2a4d176da245b9
ARG COMFYUI_VIDEOHELPERSUITE_REF=2984ec4c4b93292421888f38db74a5e8802a8ff8
ARG COMFYUI_WANVIDEOWRAPPER_REF=df8f3e49daaad117cf3090cc916c83f3d001494c
ARG COMFYUI_NVIDIA_RTX_REF=892515e3eb9a4920a131a502a047e47adca9eb0d
RUN set -eux; \
    mkdir -p /app/ComfyUI/custom_nodes; \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git /app/ComfyUI/custom_nodes/ComfyUI-Manager; \
    git -C /app/ComfyUI/custom_nodes/ComfyUI-Manager checkout "${COMFYUI_MANAGER_REF}"; \
    git clone https://github.com/city96/ComfyUI-GGUF.git /app/ComfyUI/custom_nodes/ComfyUI-GGUF; \
    git -C /app/ComfyUI/custom_nodes/ComfyUI-GGUF checkout "${COMFYUI_GGUF_REF}"; \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git /app/ComfyUI/custom_nodes/ComfyUI-KJNodes; \
    git -C /app/ComfyUI/custom_nodes/ComfyUI-KJNodes checkout "${COMFYUI_KJNODES_REF}"; \
    git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git /app/ComfyUI/custom_nodes/ComfyUI-LTXVideo; \
    git -C /app/ComfyUI/custom_nodes/ComfyUI-LTXVideo checkout "${COMFYUI_LTXVIDEO_REF}"; \
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git /app/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler; \
    git -C /app/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler checkout "${COMFYUI_SEEDVR2_REF}"; \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /app/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite; \
    git -C /app/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite checkout "${COMFYUI_VIDEOHELPERSUITE_REF}"; \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git /app/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper; \
    git -C /app/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper checkout "${COMFYUI_WANVIDEOWRAPPER_REF}"; \
    git clone https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.git /app/ComfyUI/custom_nodes/Nvidia_RTX_Nodes_ComfyUI; \
    git -C /app/ComfyUI/custom_nodes/Nvidia_RTX_Nodes_ComfyUI checkout "${COMFYUI_NVIDIA_RTX_REF}"; \
    for requirements in /app/ComfyUI/custom_nodes/*/requirements.txt; do \
        python3 -m pip install --no-cache-dir -r "${requirements}"; \
    done

# Additional CV / ML / utility packages. Anything already pulled by ComfyUI
# requirements.txt (transformers, numpy, scipy, tqdm, psutil, aiohttp,
# kornia, Pillow, requests) is intentionally omitted.
RUN python3 -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        opencv-python-headless \
        opencv-contrib-python-headless \
        scikit-image \
        imageio \
        imageio-ffmpeg \
        diffusers \
        accelerate \
        xformers \
        insightface \
        onnxruntime-gpu \
        librosa \
        soundfile \
        pandas \
        scikit-learn \
        fastapi \
        uvicorn \
        GPUtil \
        matplotlib \
        seaborn \
        controlnet-aux \
        bitsandbytes \
        pympler

# Specialized AI / vision packages with explicit version pins. moviepy is held
# at 1.0.3 because most ComfyUI video custom nodes still rely on the 1.x API.
RUN rm -rf /usr/lib/python3/dist-packages/blinker* && \
    python3 -m pip install --no-cache-dir \
        segment-anything \
        mediapipe==0.10.33 \
        rembg==2.0.75 \
        backgroundremover==0.3.6 \
        ultralytics==8.4.41 \
        trimesh \
        moviepy==1.0.3 \
        albumentations==2.0.8 \
        huggingface-hub \
        rich \
        typer \
        click

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/app/ComfyUI
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"
# Use jemalloc for better memory allocator behavior under heavy ML workloads.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_BACKENDS_CUDNN_BENCHMARK=1
# Honor the same constraint file copied from the builder so any debugging
# pip install inside the container cannot bump torch.
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# Runtime-only system dependencies. -dev packages and toolchain are not
# carried over; opencv-python-headless ships its own bundled libraries.
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        curl \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libgoogle-perftools4 \
        libjemalloc2 \
        ffmpeg && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Bring in the prebuilt Python environment, console scripts, constraint file,
# and the ComfyUI application tree from the builder stage.
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /etc/pip-constraints.txt /etc/pip-constraints.txt
COPY --from=builder /app /app

# Non-root runtime user (UID/GID 10001 mirrors values.yaml runAsUser/runAsGroup).
RUN groupadd --system --gid 10001 comfyui && \
    useradd --system --uid 10001 --gid comfyui --shell /usr/sbin/nologin \
        --create-home --home-dir /home/comfyui comfyui && \
    chown -R comfyui:comfyui /app /home/comfyui

WORKDIR /app/ComfyUI
USER 10001:10001

EXPOSE 8188

# Liveness check hits /system_stats which returns immediately once the HTTP
# server is up, well before models finish loading.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8188/system_stats || exit 1

# Start ComfyUI bound to all interfaces on the canonical port with the
# CUDA caching allocator enabled.
CMD ["python3", "main.py", \
    "--listen", "0.0.0.0", \
    "--port", "8188", \
    "--cuda-malloc"]
