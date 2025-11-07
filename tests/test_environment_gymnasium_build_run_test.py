"""Tests for building, running, and validating Gymnasium environment builds."""

import logging

import pytest

from containerl.cli import build_run, stop_container
from containerl.interface.environment.client import (
    main as validate_environment_connection,
)

# Define environments and their build configurations
TEST_CASES = [
    pytest.param(
        "./examples/gymnasium/environments/classic_control/", id="classic_control"
    ),
    pytest.param("./examples/gymnasium/environments/box2d/", id="box2d"),
    pytest.param("./examples/gymnasium/environments/mujoco/", id="mujoco"),
    pytest.param("./examples/gymnasium/environments/atari/", id="atari"),
]


@pytest.mark.parametrize("env_folder", TEST_CASES)
def test_build_run_environment(env_folder: str) -> None:
    """Test building, running, and validating a Gymnasium environment build."""
    logger = logging.getLogger(__name__)
    # Run the build_run_test command
    image = build_run(env_folder)

    success = False
    try:
        validate_environment_connection("localhost:50051")
        success = True
    except Exception as e:
        logger.error(f"Error testing environment connection: {str(e)}")

    stop_container(image)

    assert success, "Environment connection failed"  # noqa: S101
