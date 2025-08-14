FROM python:3.11-slim

WORKDIR /app

# Install pytorch
RUN pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Copy all files from the current directory to the container
COPY anylogic/stock_management/agent/ .

# Install requirements if the file exists
RUN if [ -f requirements ]; then pip install --no-cache-dir -r requirements; fi

# Set the entry point to basic_ppo.py
ENTRYPOINT ["python", "basic_ppo.py"]
