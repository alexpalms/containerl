"""Tests for building, running, and validating generic environment builds."""

import logging

import pytest

from containerl.cli import build_run, stop_container
from containerl.interface.environment.client import (
    main as validate_environment_connection,
)

# Define environments and their build configurations
TEST_CASES = [
    pytest.param(
        "./anylogic/stock_management/environment/", id="anylogic/stock_management"
    ),
    pytest.param(
        "./generic/environments/discrete_actions/", id="generic/discrete_actions"
    ),
    pytest.param(
        "./generic/environments/multibinary_actions/", id="generic/multibinary_actions"
    ),
    pytest.param(
        "./generic/environments/multidiscrete_actions/",
        id="generic/multidiscrete_actions",
    ),
    pytest.param(
        "./generic/environments/continuous_actions/", id="generic/continuous_actions"
    ),
    pytest.param(
        "./generic/environments/continuous_action/", id="generic/continuous_action"
    ),
    pytest.param(
        "./generic/environments/discrete_actions_vectorized/",
        id="generic/discrete_actions_vectorized",
    ),
    pytest.param(
        "./generic/environments/multibinary_actions_vectorized/",
        id="generic/multibinary_actions_vectorized",
    ),
    pytest.param(
        "./generic/environments/multidiscrete_actions_vectorized/",
        id="generic/multidiscrete_actions_vectorized",
    ),
    pytest.param(
        "./generic/environments/continuous_actions_vectorized/",
        id="generic/continuous_actions_vectorized",
    ),
    pytest.param(
        "./generic/environments/continuous_action_vectorized/",
        id="generic/continuous_action_vectorized",
    ),
]


@pytest.mark.parametrize("env_folder", TEST_CASES)
def test_build_run_environment(env_folder: str) -> None:
    """Test building, running, and validating a generic environment build."""
    # Run the build_run_test command
    image = build_run(env_folder)
    logger = logging.getLogger(__name__)

    success = False
    try:
        validate_environment_connection("localhost:50051")
        success = True
    except Exception as e:
        logger.error(f"Error testing environment connection: {str(e)}")

    stop_container(image)

    assert success, "Environment connection failed"  # noqa: S101
