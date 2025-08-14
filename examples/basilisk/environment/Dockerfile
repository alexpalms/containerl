FROM python:3.11-slim

WORKDIR /app

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
RUN pip install pandas numpy matplotlib scipy pytest pytest-cov pytest-xdist bokeh protobuf tqdm

# Install dependencies from requirements file
RUN pip install -r requirements_dev.txt

# Run conanfile.py
RUN python conanfile.py

WORKDIR /app

RUN git clone https://github.com/AVSLab/bsk_rl.git

WORKDIR /app/bsk_rl

RUN pip install -e .

RUN finish_install

# Copy all files from the current directory to the container
COPY basilisk/environment/ .

# Install requirements if the file exists
RUN if [ -f requirements ]; then pip install --no-cache-dir -r requirements; fi

ENTRYPOINT ["python", "environment.py"]

