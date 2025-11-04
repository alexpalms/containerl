FROM python:3.11-slim

WORKDIR /app

RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*;

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

RUN uv venv

# Copy full repository, build/install package with `uv`, then remove repo
COPY . /tmp/repo

RUN set -eux; \
    cd /tmp/repo && uv build; \
    cd /app; \
    uv pip install --no-cache-dir /tmp/repo/dist/*.whl; \
    rm -rf /tmp/repo

# 2) Copy only the example folder needed at runtime
COPY examples/generic/agents/continuous_action/ .

# Install example-specific requirements if present
RUN if [ -f requirements.txt ]; then uv pip install --no-cache-dir -r requirements.txt; fi

ENTRYPOINT ["uv", "run", "python", "agent.py"]
