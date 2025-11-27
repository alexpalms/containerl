FROM python:3.12-slim

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

# Copy all files from the current directory to the container
COPY examples/anylogic/stock_management/proportional_agent/ .

# Install requirements if the file exists
RUN if [ -f requirements.txt ]; then uv pip install --no-cache-dir -r requirements.txt; fi

# Set the entry point to basic_ppo.py
ENTRYPOINT ["uv", "run", "python", "agent.py"]
