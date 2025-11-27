"""Tests for building, running, and validating generic agent environments."""

import logging
from typing import Any

import pytest

from containerl import agent_check
from containerl.cli import build_run, stop_container

# Test cases with descriptive IDs
TEST_CASES = [
    pytest.param(
        "./examples/anylogic/stock_management/agent/",
        {
            "port": "localhost:50051",
            "init_args": {
                "model_name": "model.zip",
                "device": "cpu",
            },
        },
        id="anylogic/stock_management/rl_agent",
    ),
    pytest.param(
        "./examples/anylogic/stock_management/proportional_agent/",
        {
            "port": "localhost:50051",
            "init_args": {
                "target_stock": 5000,
                "proportional_constant": 0.1,
            },
        },
        id="anylogic/stock_management/proportional_agent",
    ),
    pytest.param(
        "./examples/generic/agents/discrete_actions/",
        {
            "port": "localhost:50051",
            "init_args": {},
        },
        id="generic/discrete_actions",
    ),
    pytest.param(
        "./examples/generic/agents/multibinary_actions/",
        {
            "port": "localhost:50051",
            "init_args": {},
        },
        id="generic/multibinary_actions",
    ),
    pytest.param(
        "./examples/generic/agents/multidiscrete_actions/",
        {
            "port": "localhost:50051",
            "init_args": {},
        },
        id="generic/multidiscrete_actions",
    ),
    pytest.param(
        "./examples/generic/agents/continuous_actions/",
        {
            "port": "localhost:50051",
            "init_args": {},
        },
        id="generic/continuous_actions",
    ),
    pytest.param(
        "./examples/generic/agents/continuous_action/",
        {
            "port": "localhost:50051",
            "init_args": {},
        },
        id="generic/single_continuous_action",
    ),
]


@pytest.mark.parametrize("env_folder, env_param", TEST_CASES)
def test_build_run_agent(env_folder: str, env_param: dict[str, Any]) -> None:
    """Test building, running, and validating a generic agent environment."""
    logger = logging.getLogger(__name__)
    image = build_run(env_folder)

    success = False
    try:
        init_args: dict[str, Any] = env_param.get("init_args", {})
        agent_check(env_param["port"], **init_args)
        success = True
    except Exception as e:
        logger.error(f"Error testing agent connection: {str(e)}")

    stop_container(image)

    assert success, "Agent connection failed"  # noqa: S101
