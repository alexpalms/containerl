FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all files from the current directory to the container
COPY gymnasium/environments/box2d/ .

# Install requirements if the file exists
RUN if [ -f requirements ]; then pip install --no-cache-dir -r requirements; fi

# Set the entry point to environment.py
ENTRYPOINT ["python", "environment.py"]
