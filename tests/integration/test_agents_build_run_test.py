"""Tests for building, running, and validating generic agent environments."""

import logging
import os
from typing import Any

import pytest

from containerl import agent_check
from containerl.cli import build_run, stop_container

pytestmark = pytest.mark.integration

current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.join(current_directory, "../..")

# Test cases with descriptive IDs
AGENTS_BASE_FOLDER = "examples/agents/"
TEST_CASES = [
    pytest.param(
        "./anylogic/stock_management/deep_rl/",
        {
            "model_name": "model.zip",
            "device": "cpu",
        },
        id="anylogic/deep_rl",
    ),
    pytest.param(
        "./anylogic/stock_management/proportional/",
        {
            "target_stock": 5000,
            "proportional_constant": 0.1,
        },
        id="anylogic/proportional",
    ),
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
        id="generic/single_continuous_action",
    ),
]


@pytest.mark.parametrize("agent_folder, init_args", TEST_CASES)
def test_build_run_agent(agent_folder: str, init_args: dict[str, Any]) -> None:
    """Test building, running, and validating a generic agent environment."""
    logger = logging.getLogger(__name__)
    image, addresses = build_run(
        os.path.join(root_directory, AGENTS_BASE_FOLDER),
        entrypoint_args=[f"{agent_folder}/agent.py"],
    )
    success = False
    try:
        agent_check(addresses[0], **init_args)
        success = True
    except Exception as e:
        logger.error(f"Error testing agent connection: {str(e)}")

    stop_container(image)

    assert success, "Agent connection failed"  # noqa: S101
