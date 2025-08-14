FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgl1-mesa-glx \
    libegl1 \
    libxrender1 \
    libglib2.0-0 \
    ffmpeg \
    libgles2 \
    libglvnd0 \
    libglx0 \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 \ 
    nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 torch torchaudio torchvision \ 
    && pip cache purge && conda clean --all

WORKDIR /app

# Copy all files from the current directory to the container
COPY genesis/locomotion/environment/ .

# Install pytorch
RUN pip uninstall -y torch torchaudio torchvision && pip install torch==2.5.1+cpu --no-cache-dir --index-url https://download.pytorch.org/whl/cpu

# Install requirements if the file exists
RUN if [ -f requirements ]; then pip install --no-cache-dir -r requirements; fi

# Set the entry point to environment.py
ENTRYPOINT ["python", "environment.py"]
