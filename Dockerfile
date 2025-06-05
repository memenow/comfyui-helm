# ComfyUI Docker Image
# ====================
# 
# This Dockerfile builds a production-ready container for ComfyUI, a powerful and modular
# Stable Diffusion GUI and backend. The image includes CUDA support for GPU acceleration,
# comprehensive machine learning libraries, and all necessary dependencies.
#
# Base Image: NVIDIA CUDA 12.4.1 with cuDNN on Ubuntu 22.04
# Target: GPU-accelerated machine learning workloads
# Architecture: x86_64
#
# Build command:
#   docker build -t comfyui:latest .
#
# Run command:
#   docker run -d --gpus all -p 8188:8188 comfyui:latest

# Use NVIDIA's official CUDA runtime image with cuDNN for deep learning acceleration
# This provides CUDA 12.4.1, cuDNN, and Ubuntu 22.04 LTS as the base system
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Configure environment variables for optimal container behavior
# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
# Ensure Python output is sent straight to terminal without buffering
ENV PYTHONUNBUFFERED=1
# Set CUDA installation directory
ENV CUDA_HOME=/usr/local/cuda
# Add CUDA binaries to system PATH
ENV PATH=${CUDA_HOME}/bin:${PATH}
# Add CUDA libraries to library path
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies and development tools
# This comprehensive package list includes:
# - Python 3 and development headers
# - Build tools for compiling native extensions
# - Media processing libraries (OpenCV, FFmpeg)
# - Graphics and visualization libraries
# - Performance optimization libraries (jemalloc, perftools)
# - SSL/TLS and cryptographic libraries
# - System monitoring tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        git \
        wget \
        curl \
        unzip \
        build-essential \
        cmake \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libgoogle-perftools4 \
        libjemalloc2 \
        libopencv-dev \
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
        ffmpeg \
        nvtop && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Set the working directory for the application
# All subsequent commands will run from this directory
WORKDIR /app

# Upgrade Python package management tools to latest versions
# This ensures compatibility with modern Python packages and security updates
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
# PyTorch is the core deep learning framework used by ComfyUI
# The specific versions are chosen for stability and CUDA compatibility
RUN pip3 install --no-cache-dir \
    torch==2.5.0 \
    torchvision==0.20.0 \
    torchaudio==2.5.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Clone the ComfyUI repository from GitHub
# This gets the latest stable version of the ComfyUI application
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# Change to the ComfyUI directory for subsequent operations
WORKDIR /app/ComfyUI

# Install ComfyUI's core dependencies from requirements.txt
# This includes the essential packages needed for basic functionality
RUN pip3 install --no-cache-dir -r requirements.txt

# Install core computer vision and image processing libraries
# These packages provide essential functionality for image manipulation,
# computer vision tasks, and deep learning model inference
RUN pip3 install --no-cache-dir \
    opencv-python-headless \
    opencv-contrib-python-headless \
    scikit-image \
    imageio \
    imageio-ffmpeg \
    diffusers \
    transformers \
    accelerate \
    xformers \
    insightface \
    onnxruntime-gpu \
    librosa \
    soundfile \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    requests \
    aiohttp \
    fastapi \
    uvicorn \
    pillow-simd \
    tqdm \
    psutil \
    GPUtil \
    matplotlib \
    seaborn \
    controlnet-aux \
    bitsandbytes \
    pympler

# Install additional specialized AI and computer vision packages
# These provide advanced functionality for specific use cases:
# - segment-anything: Meta's SAM for object segmentation
# - mediapipe: Google's ML framework for perception tasks
# - rembg & backgroundremover: Background removal tools
# - ultralytics: YOLO object detection framework
# - trimesh: 3D mesh processing
# - moviepy: Video editing and processing
# - kornia: Computer vision library for PyTorch
# - albumentations: Advanced image augmentation
# - huggingface-hub: Access to Hugging Face model repository
# - rich, typer, click: Enhanced CLI and terminal output
RUN pip3 install --no-cache-dir \
    segment-anything \
    mediapipe \
    rembg \
    backgroundremover \
    ultralytics \
    trimesh \
    moviepy \
    kornia \
    albumentations \
    huggingface-hub \
    rich \
    typer \
    click
# Set appropriate file permissions for the ComfyUI application directory
# This ensures the application can read its files and execute properly
RUN chmod -R 755 /app/ComfyUI

# Configure runtime environment variables for optimal performance
# LD_PRELOAD: Use jemalloc for better memory allocation performance
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
# Disable CUDA launch blocking for better asynchronous GPU operations
ENV CUDA_LAUNCH_BLOCKING=0
# Enable cuDNN benchmarking for optimal performance with consistent input sizes
ENV TORCH_BACKENDS_CUDNN_BENCHMARK=1

# Expose port 8188 for ComfyUI web interface
# This is the default port that ComfyUI uses for its web-based GUI
EXPOSE 8188

# Configure health check to monitor container status
# This checks if the ComfyUI web interface is responding properly
# - Check interval: every 30 seconds
# - Timeout: 10 seconds per check
# - Start period: 60 seconds (allows time for startup)
# - Retries: 3 failed checks before marking as unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8188/ || exit 1

# Start ComfyUI with production-ready configuration
# - Listen on all interfaces (0.0.0.0) for container accessibility
# - Use port 8188 (standard ComfyUI port)
# - Enable CUDA memory allocation optimizations
CMD ["python3", "main.py", \
    "--listen", "0.0.0.0", \
    "--port", "8188", \
    "--cuda-malloc"]