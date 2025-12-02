<!-- <p align="center">
<a href="https://alexpalms.github.io/projects/02-rl_cuas/"><img src="https://img.shields.io/badge/blog-read%20post-blue" alt="Blog Post"/></a>
</p> -->

<p align="center">
<a href="https://github.com/alexpalms/containerl/tags"><img src="https://img.shields.io/github/v/tag/alexpalms/containerl?label=latest%20tag&logo=github" alt="Latest Tag"/></a>
<a href="https://pypi.org/project/containerl/"><img src="https://img.shields.io/pypi/v/containerl?logo=pypi" alt="Pypi version"/></a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/type%20checking-pyright-2A6DB0?logo=python&logoColor=white" alt="Type Hints"/>
<img src="https://img.shields.io/badge/linting-ruff-4B8BBE?logo=python&logoColor=white" alt="Code Formatting"/>
<img src="https://img.shields.io/badge/testing-pytest-2A6DB0?logo=python&logoColor=white" alt="Pytest"/>
</p>
<p align="center">
<a href="https://github.com/alexpalms/containerl/actions/workflows/type-hints-check.yaml"><img src="https://img.shields.io/github/actions/workflow/status/alexpalms/containerl/type-hints-check.yaml?label=type%20hints&logo=github" alt="Type Hints"/></a>
<a href="https://github.com/alexpalms/containerl/actions/workflows/code-formatting-check.yaml"><img src="https://img.shields.io/github/actions/workflow/status/alexpalms/containerl/code-formatting-check.yaml?label=code%20formatting&logo=github" alt="Code Formatting"/></a>
<a href="https://github.com/alexpalms/containerl/actions/workflows/pytest.yaml"><img src="https://img.shields.io/github/actions/workflow/status/alexpalms/containerl/pytest.yaml?label=pytest&logo=github" alt="Pytest"/></a>
<a href="https://codecov.io/github/alexpalms/containerl"><img src="https://codecov.io/github/alexpalms/containerl/graph/badge.svg?token=4817P3HFDN" alt="PytestCoverage"/></a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/supported%20os-linux/osx/win-blue" alt="Supported OS"/>
<img src="https://img.shields.io/badge/python-%3E%3D3.12-blue" alt="Python Version"/>
<img src="https://img.shields.io/github/last-commit/alexpalms/containerl/main?label=repo%20latest%20update&logo=readthedocs" alt="Latest Repo Update"/>
</p>
<p align="center">
<img src="https://img.shields.io/github/license/alexpalms/containerl?cacheBust=1" alt="License"/>
</p>

# ContaineRL

Containerize your RL Environments and Agents


## Overview

ContaineRL is a toolkit to package and deploy reinforcement-learning (RL) environments and agents inside reproducible containers. It provides a compact Python API and a command-line interface (entry point: `containerl-cli`) to manage environment/agent lifecycles, build artifacts, and integrate with gRPC/msgpack-based interfaces.

## Project layout

- src/
  - containerl/ (package)
    - cli.py            # CLI entry point (containerl.cli:main)
    - interface/        # Proto/gRPC bindings and transport abstractions
    - environment/      # Environment adapters and helpers
    - agent/            # Agent runners and integration code
- tests/
  - unit/              # Fast, isolated unit tests (no external services)
  - integration/       # Slower tests that exercise containers, gRPC, networks
- examples/            # Example agents and environments


## Installation

Install for development:

- Python 3.12+ is required, [UV](https://docs.astral.sh/uv/) is strongly suggested.
- Clone and install editable:

```bash
uv sync --dev
```

This installs the `containerl-cli` console script (defined in pyproject.toml) and dev tools (pyright, pytest, ruff, etc.).


## Quickstart (Python Package)

ContaineRL provides simple abstractions to expose RL environments and agents as containerized services.

### Exposing an Environment

Wrap your environment class (Gymnasium-compatible) and expose it via gRPC:

```python
import gymnasium as gym
from containerl import AllowedTypes, AllowedInfoValueTypes, create_environment_server

class Environment(gym.Env):
    def __init__(self, render_mode: str, env_name: str):
        self._env = gym.make(env_name, render_mode=render_mode)
        self.observation_space = gym.spaces.Dict({"observation": self._env.observation_space})
        self.action_space = self._env.action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return {"observation": obs}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return {"observation": obs}, float(reward), terminated, truncated, info

if __name__ == "__main__":
    create_environment_server(Environment)
```

Run this script as your Docker container entrypoint. The server listens on port 50051 by default.

### Exposing an Agent

Implement the `CRLAgent` interface and expose it via gRPC:

```python
import numpy as np
from gymnasium import spaces
from containerl import CRLAgent, AllowedTypes, create_agent_server

class Agent(CRLAgent):
    def __init__(self, target: float, gain: float):
        self.target = target
        self.gain = gain
        self.observation_space = spaces.Dict({"state": spaces.Box(0, 100, shape=(1,))})
        self.action_space = spaces.Box(0, 10, shape=(1,), dtype=np.float32)

    def get_action(self, observation: dict[str, AllowedTypes]) -> AllowedTypes:
        return np.clip(self.gain * (self.target - observation["state"]), 0, 10)

if __name__ == "__main__":
    create_agent_server(Agent)
```

Both `create_environment_server` and `create_agent_server` handle initialization arguments passed via gRPC, spawn the service, and manage the lifecycle automatically.

See more examples provided with the repository in the [examples/](examples/) directory.


## Quickstart (CLI)

Show help and global options:

```bash
uv run containerl-cli --help
```

Common, supported commands (see `--help` for full options):

### Build a Docker image from a directory containing a Dockerfile:

```bash
uv run containerl-cli build ./examples/gymnasium/environments/atari/ -n my-image -t v1
```

### Run a built image (maps container port 50051 to host by default):

```bash
# Run with explicit image name
uv run containerl-cli run my-image:v1

# Run with a custom container name (only when count=1)
uv run containerl-cli run my-image:v1 --name my-container

# Run multiple containers
uv run containerl-cli run my-image:v1 --count 3

# Run in interactive mode (only when count=1)
uv run containerl-cli run my-image:v1 -i
```

### Test connection to a running container:

```bash
# Test with initialization arguments
uv run containerl-cli test --address localhost:50051 \
  --init-arg render_mode="rgb_array" \
  --init-arg env_name="ALE/Breakout-v5" \
  --init-arg obs_type="ram"
```

### Build an image and run containers from it:

```bash
uv run containerl-cli build-run ./examples/gymnasium/environments/atari/

# With a custom container name
uv run containerl-cli build-run ./examples/gymnasium/environments/atari/ --container-name my-env
```

### Build, run and test a container (invokes client checks):

```bash
# With initialization arguments (supports int, float, bool, and string values)
uv run containerl-cli build-run-test ./examples/gymnasium/environments/atari/ \
  --init-arg render_mode="rgb_array" \
  --init-arg env_name="ALE/Breakout-v5" \
  --init-arg obs_type="ram"
```

### Stop containers by image or by name:

```bash
# Stop all containers started from a given image
uv run containerl-cli stop --image my-image:v1

# Stop container(s) by name
uv run containerl-cli stop --name my-container
```

The CLI subcommands implemented are: `build`, `run`, `test`, `stop`, `build-run`, and `build-run-test`. Use `containerl-cli <command> --help` for command-specific flags.

**Important notes:**
- The default image name is `containerl-build:latest` (used when no name is specified in `build` or `run` commands)
- Container naming (`--name` for `run`, `--container-name` for `build-run`/`build-run-test`), volume mounting (`--volume`), interactive mode (`-i`), and attach mode (`-a`) are only available when `--count 1` (the default)
- The `stop` command requires either `--image` or `--name` but not both
- Initialization arguments (`--init-arg key=value`) can be passed to `test` and `build-run-test` commands to configure the environment or agent. Multiple init args can be specified, and values are automatically converted to int, float, bool, or string types

## Testing strategy (unit vs integration)

The repository separates tests into two folders: `tests/unit/` and `tests/integration/` to speed up the inner development loop and to make CI scheduling simpler.

### Unit tests: fast, deterministic, no network or container dependencies. Run quickly on every commit:

```bash
pytest tests/unit
```

### Integration tests: exercise containers, gRPC interfaces, or external services. Run them less frequently or in dedicated CI jobs:

```bash
pytest tests/integration
```

## License & Contact

This project is MIT licensed. For questions or issues open an issue at the repository or contact the maintainer listed in pyproject.toml.

Containerize your RL Environments and Agents
