"""Tests for building, running, and validating Gymnasium environment builds."""

import logging
from typing import Any

import pytest

from containerl import environment_check, gym_environment_check
from containerl.cli import build_run, stop_container

# Define environments and their build configurations
TEST_CASES = [
    pytest.param(
        "./examples/gymnasium/environments/classic_control/",
        {
            "port": "localhost:50051",
            "render_mode": "rgb_array",
            "init_args": {
                "env_name": "CartPole-v1",
            },
        },
        id="classic_control",
    ),
    pytest.param(
        "./examples/gymnasium/environments/box2d/",
        {
            "port": "localhost:50051",
            "render_mode": "rgb_array",
            "init_args": {
                "env_name": "LunarLander-v3",
            },
        },
        id="box2d",
    ),
    pytest.param(
        "./examples/gymnasium/environments/mujoco/",
        {
            "port": "localhost:50051",
            "render_mode": "rgb_array",
            "init_args": {
                "env_name": "Ant-v5",
            },
        },
        id="mujoco",
    ),
    pytest.param(
        "./examples/gymnasium/environments/atari/",
        {
            "port": "localhost:50051",
            "render_mode": "rgb_array",
            "init_args": {
                "env_name": "ALE/Breakout-v5",
                "obs_type": "ram",
            },
        },
        id="atari",
    ),
]


@pytest.mark.parametrize("env_folder, env_param", TEST_CASES)
def test_build_run_environment(env_folder: str, env_param: dict[str, Any]) -> None:
    """Test building, running, and validating a Gymnasium environment build."""
    logger = logging.getLogger(__name__)
    # Run the build_run_test command
    image = build_run(env_folder)

    success = False
    try:
        environment_check(
            env_param["port"],
            render_mode=env_param["render_mode"],
            **env_param["init_args"],
        )
        gym_environment_check(
            env_param["port"],
            render_mode=env_param["render_mode"],
            **env_param["init_args"],
        )
        success = True
    except Exception as e:
        logger.error(f"Error testing environment connection: {str(e)}")

    # Ensure container is stopped using the image identifier
    stop_container(str(image))

    assert success, "Environment connection failed"  # noqa: S101
