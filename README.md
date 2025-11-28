# ContaineRL

Containerize your RL Environments and Agents


## Overview

ContaineRL is a CLI-first toolkit to package, run, and test reinforcement-learning (RL) environments and agents inside reproducible containers. It provides a compact Python API and a command-line interface (entry point: `containerl-cli`) to manage environment/agent lifecycles, build artifacts, and integrate with gRPC/msgpack-based interfaces.

_Last updated: 2025-11-28_

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
- docs/                # Sphinx documentation source
- stubs/               # Type stubs used for strict typing


## Installation

Install for development:

- Python 3.12+ is required.
- Clone and install editable:

```bash
pip install -e .[dev]
```

This installs the `containerl-cli` console script (defined in pyproject.toml) and dev tools (mypy, pytest, ruff, etc.).


## Quickstart (CLI)

Show help and global options:

```bash
containerl-cli --help
```

Common usage patterns (commands are illustrative; run `--help` for exact options):

- Build a containerized environment or agent:

```bash
containerl-cli build --target env --path examples/cartpole
```

- Run an environment with a local agent:

```bash
containerl-cli run --env CartPole-v1 --agent ./agents/ppo
```

- List available examples or registered environments:

```bash
containerl-cli list --examples
```

The CLI is intentionally thin: prefer `--help` for the canonical flags and subcommands implemented in `containerl.cli`.


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
- Run static typing with mypy/pyright (project is configured for strict checks).
- Use `examples/` as integration smoke tests for local development.

Common commands:

```bash
# run unit tests with coverage
pytest tests/unit --cov=containerl

# run lint
ruff check src tests

# run type checks
mypy
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
