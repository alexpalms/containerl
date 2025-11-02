"""Tests for building, running, and validating generic agent environments."""

import logging

import pytest

from containerl.cli import build_run, stop_container
from containerl.interface.agent.client import main as validate_agent_connection

# Test cases with descriptive IDs
TEST_CASES = [
    pytest.param("./anylogic/stock_management/agent/", id="anylogic/stock_management"),
    pytest.param("./generic/agents/discrete_actions/", id="generic/discrete_actions"),
    pytest.param(
        "./generic/agents/multibinary_actions/", id="generic/multibinary_actions"
    ),
    pytest.param(
        "./generic/agents/multidiscrete_actions/", id="generic/multidiscrete_actions"
    ),
    pytest.param(
        "./generic/agents/continuous_actions/", id="generic/continuous_actions"
    ),
    pytest.param(
        "./generic/agents/continuous_action/", id="generic/single_continuous_action"
    ),
]


@pytest.mark.parametrize("env_folder", TEST_CASES)
def test_build_run_agent(env_folder: str) -> None:
    """Test building, running, and validating a generic agent environment."""
    logger = logging.getLogger(__name__)
    image = build_run(env_folder)

    success = False
    try:
        validate_agent_connection("localhost:50051")
        success = True
    except Exception as e:
        logger.error(f"Error testing agent connection: {str(e)}")

    stop_container(image)

    assert success, "Agent connection failed"  # noqa: S101
