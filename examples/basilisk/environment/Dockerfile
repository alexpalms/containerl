FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN uv venv

# Copy full repository, build/install package with `uv`, then remove repo
COPY . /tmp/repo

RUN set -eux; \
    cd /tmp/repo && uv build; \
    cd /app; \
    uv pip install --no-cache-dir /tmp/repo/dist/*.whl; \
    rm -rf /tmp/repo

RUN apt-get update && apt-get install -y \
    git \
    swig \
    libgtk2.0 \
    && rm -rf /var/lib/apt/lists/*

# Clone the specific v2.7.0 tag of Basilisk
RUN git clone -b v2.7.0 https://github.com/AVSLab/basilisk.git

# Change working directory to basilisk
WORKDIR /app/basilisk

# Install common Python dependencies that Basilisk requires
RUN uv pip install pandas numpy matplotlib scipy pytest pytest-cov pytest-xdist bokeh protobuf tqdm

# Install dependencies from requirements file
RUN uv pip install -r requirements_dev.txt

# Run conanfile.py
RUN python conanfile.py

WORKDIR /app

RUN git clone https://github.com/AVSLab/bsk_rl.git

WORKDIR /app/bsk_rl

RUN uv pip install -e .

RUN finish_install

# Copy all files from the current directory to the container
COPY basilisk/environment/ .

ENTRYPOINT ["uv", "run", "python", "environment.py"]

