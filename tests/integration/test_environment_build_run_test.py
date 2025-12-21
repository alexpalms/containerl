"""Tests for building, running, and validating generic environment builds."""

import logging
import os
from typing import Any

import pytest

from containerl import (
    environment_check,
    gym_environment_check,
    vec_environment_check,
)
from containerl.cli import build_run, stop_container

current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.join(current_directory, "../..")

pytestmark = pytest.mark.integration

# Define environments and their build configurations
ENVS_BASE_FOLDER = "examples/environments/"
TEST_CASES = [
    pytest.param(
        "./generic/discrete_actions/",
        {},
        id="generic/discrete_actions",
    ),
    pytest.param(
        "./generic/multibinary_actions/",
        {},
        id="generic/multibinary_actions",
    ),
    pytest.param(
        "./generic/multidiscrete_actions/",
        {},
        id="generic/multidiscrete_actions",
    ),
    pytest.param(
        "./generic/continuous_actions/",
        {},
        id="generic/continuous_actions",
    ),
    pytest.param(
        "./generic/continuous_action/",
        {},
        id="generic/continuous_action",
    ),
    pytest.param(
        "./generic/discrete_actions_vectorized/",
        {},
        id="generic/discrete_actions_vectorized",
    ),
    pytest.param(
        "./generic/multibinary_actions_vectorized/",
        {},
        id="generic/multibinary_actions_vectorized",
    ),
    pytest.param(
        "./generic/multidiscrete_actions_vectorized/",
        {},
        id="generic/multidiscrete_actions_vectorized",
    ),
    pytest.param(
        "./generic/continuous_actions_vectorized/",
        {},
        id="generic/continuous_actions_vectorized",
    ),
    pytest.param(
        "./generic/continuous_action_vectorized/",
        {},
        id="generic/continuous_action_vectorized",
    ),
    pytest.param(
        "./gymnasium/classic_control/",
        {
            "render_mode": None,
            "env_name": "CartPole-v1",
        },
        id="gymnasium/classic_control",
    ),
    pytest.param(
        "./gymnasium/box2d/",
        {
            "render_mode": "rgb_array",
            "env_name": "LunarLander-v3",
        },
        id="gymnasium/box2d",
    ),
    pytest.param(
        "./gymnasium/mujoco/",
        {
            "render_mode": "rgb_array",
            "env_name": "Ant-v5",
        },
        id="gymnasium/mujoco",
    ),
    pytest.param(
        "./gymnasium/atari/",
        {
            "render_mode": None,
            "env_name": "ALE/Breakout-v5",
            "obs_type": "ram",
        },
        id="gymnasium/atari",
    ),
]


@pytest.mark.parametrize("env_folder, init_args", TEST_CASES)
def test_build_run_environment(env_folder: str, init_args: dict[str, Any]) -> None:
    """Test building, running, and validating a generic environment build."""
    # Run the build_run_test command
    logger = logging.getLogger(__name__)
    image, addresses = build_run(
        os.path.join(root_directory, ENVS_BASE_FOLDER),
        entrypoint_args=[f"{env_folder}/environment.py"],
    )

    success = False
    try:
        if "vectorized" in env_folder:
            vec_environment_check(addresses[0], **init_args)
        else:
            environment_check(addresses[0], **init_args)
            gym_environment_check(addresses[0], **init_args)
        success = True
    except Exception as e:
        logger.error(f"Error testing environment connection: {str(e)}")

    # Ensure container is stopped using the image identifier
    stop_container(image)

    assert success, "Environment connection failed"  # noqa: S101
