# ContaineRL

Containerize your RL Environments and Agents


## Overview

ContaineRL is a CLI-first toolkit to package, run, and test reinforcement-learning (RL) environments and agents inside reproducible containers. It provides a compact Python API and a command-line interface (entry point: `containerl-cli`) to manage environment/agent lifecycles, build artifacts, and integrate with gRPC/msgpack-based interfaces.

_Last updated: 2025-11-30_

## Project layout

- src/
  - containerl/ (package)
    - cli.py            # CLI entry point (containerl.cli:main)
    - interface/        # Proto/gRPC bindings and transport abstractions
    - env/              # Environment adapters and helpers
    - agent/            # Agent runners and integration code
    - core/             # Core primitives and shared utilities
- tests/
  - unit/              # Fast, isolated unit tests (no external services)
  - integration/       # Slower tests that exercise containers, gRPC, networks
- examples/            # Example agents and environments
- stubs/               # Type stubs used for strict typing


## Installation

Install for development:

- Python 3.12+ is required.
- Clone and install editable:

```bash
uv sync
```

This installs the `containerl-cli` console script (defined in pyproject.toml) and dev tools (pyright, pytest, ruff, etc.).


## Quickstart (CLI)

Show help and global options:

```bash
uv run containerl-cli --help
```

Common, supported commands (see `--help` for full options):

- Build a Docker image from a directory containing a Dockerfile:

```bash
uv run containerl-cli build ./examples/gymnasium/environments/atari/ -n my-image -t v1 [-v]
```

- Run a built image (maps container port 50051 to host by default):

```bash
# Run with explicit image name
uv run containerl-cli run my-image:v1 [-v]

# Run with a custom container name (only when count=1)
uv run containerl-cli run my-image:v1 --name my-container [-v]

# Run multiple containers (volumes, interactive, attach, and naming only work with count=1)
uv run containerl-cli run my-image:v1 --count 3

# Run in interactive mode (only when count=1)
uv run containerl-cli run my-image:v1 -i
```

- Test connection to a running container:

```bash
# Test with initialization arguments
uv run containerl-cli test --address localhost:50051 \
  --init-arg render_mode="rgb_array" \
  --init-arg env_name="ALE/Breakout-v5" \
  --init-arg obs_type="ram"
```

- Build an image and run containers from it:

```bash
uv run containerl-cli build-run ./examples/gymnasium/environments/atari/ [-v]

# With a custom container name
uv run containerl-cli build-run ./examples/gymnasium/environments/atari/ --container-name my-env [-v]
```

- Build, run and test a container (invokes client checks):

```bash
# With initialization arguments (supports int, float, bool, and string values)
uv run containerl-cli build-run-test ./examples/gymnasium/environments/atari/ \
  --init-arg render_mode="rgb_array" \
  --init-arg env_name="ALE/Breakout-v5" \
  --init-arg obs_type="ram"
```

- Stop containers by image or by name:

```bash
# Stop all containers started from a given image
uv run containerl-cli stop --image my-image:v1 [-v]

# Stop container(s) by name
uv run containerl-cli stop --name my-container [-v]
```

The CLI subcommands implemented are: `build`, `run`, `test`, `stop`, `build-run`, and `build-run-test`. Use `containerl-cli <command> --help` for command-specific flags.

**Important notes:**
- The default image name is `containerl-build:latest` (used when no name is specified in `build` or `run` commands)
- Container naming (`--name` for `run`, `--container-name` for `build-run`/`build-run-test`), volume mounting (`--volume`), interactive mode (`-i`), and attach mode (`-a`) are only available when `--count 1` (the default)
- The `stop` command requires either `--image` or `--name` but not both
- Initialization arguments (`--init-arg key=value`) can be passed to `test` and `build-run-test` commands to configure the environment or agent. Multiple init args can be specified, and values are automatically converted to int, float, bool, or string types


## Configuration and Extensibility

- Configuration is read from environment variables and optional YAML/JSON files passed to commands.
- The interface layer sits between CLI commands and runtime transports (gRPC, local IPC). Implement a new transport by adding an adapter in `src/containerl/interface`.
- Agents should implement the runtime contract defined in `src/containerl/agent` (see examples for reference).


## Testing strategy (unit vs integration)

The repository separates tests into two folders: `tests/unit/` and `tests/integration/` to speed up the inner development loop and to make CI scheduling simpler.

- Unit tests: fast, deterministic, no network or container dependencies. Run quickly on every commit:

```bash
pytest tests/unit
```

- Integration tests: exercise containers, gRPC interfaces, or external services. Run them less frequently or in dedicated CI jobs:

```bash
pytest tests/integration
# or use a marker if configured
pytest -m integration
```

Recommendations:
- Keep unit tests focused on pure logic and small adapter behavior.
- Use pytest markers (e.g., `@pytest.mark.integration`) to tag slow tests and register the marker in `pytest.ini`.
- In CI, run unit tests on every push and run integration tests in a scheduled pipeline or gated job.


## Development workflow

- Format and lint with ruff and pre-commit to keep style consistent.
- Run static typing with pyright (project is configured for strict checks).
- Use `examples/` as integration smoke tests for local development.

Common commands:

```bash
# run unit tests with coverage
pytest tests/unit --cov=containerl

# run lint
ruff check src tests

# run type checks
uv run pyright
```


## Contributing

- Fork the repository and open a branch for your change.
- Keep changes small and focused; add unit tests for new logic and integration tests when external behavior changes.
- Run pre-commit and type checks before opening a PR.


## Troubleshooting

- If a CLI command fails, re-run with `--debug` or check `~/.containerl/logs` (if present) for runtime traces.
- For integration issues, ensure local container runtime and network resources are available and that gRPC ports do not conflict.


## License & Contact

This project is MIT licensed. For questions or issues open an issue at the repository or contact the maintainer listed in pyproject.toml.

Containerize your RL Environments and Agents
