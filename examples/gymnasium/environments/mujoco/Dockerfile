FROM python:3.11-slim

WORKDIR /app

# Copy all files from the current directory to the container
COPY gymnasium/environments/mujoco/ .

RUN apt-get update && apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa

# Install requirements if the file exists
RUN if [ -f requirements ]; then pip install --no-cache-dir -r requirements; fi

# Set the entry point to environment.py
ENTRYPOINT ["python", "environment.py"]