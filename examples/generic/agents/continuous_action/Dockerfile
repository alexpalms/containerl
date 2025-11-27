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

# Copy only the example folder needed at runtime
COPY examples/generic/agents/continuous_action/ .

# Install example-specific requirements if present
RUN if [ -f requirements.txt ]; then uv pip install --no-cache-dir -r requirements.txt; fi

ENTRYPOINT ["uv", "run", "python", "agent.py"]
