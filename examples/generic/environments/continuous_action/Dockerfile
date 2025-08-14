FROM python:3.11-slim

WORKDIR /app

# Copy all files from the current directory to the container
COPY generic/environments/continuous_action/ .

# Install requirements if the file exists
RUN if [ -f requirements ]; then pip install --no-cache-dir -r requirements; fi

# Set the entry point to environment.py
ENTRYPOINT ["python", "environment.py"]
