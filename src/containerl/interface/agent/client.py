"""Client for connecting to a remote agent via gRPC."""

import logging

import grpc
import msgpack
from gymnasium import spaces

from ..proto_pb2 import AgentInitRequest, ObservationRequest

# Add the interface directory to the path to import the generated gRPC code
from ..proto_pb2_grpc import AgentServiceStub
from ..utils import (
    AllowedInfoValueTypes,
    AllowedSerializableTypes,
    AllowedTypes,
    native_to_numpy,
    native_to_numpy_space,
    numpy_to_native,
)
from .server_factory import CRLAgent


class CRLAgentClient(CRLAgent):
    """Client for connecting to a remote agent via gRPC."""

    def __init__(
        self,
        server_address: str,
        timeout: float = 60.0,
        **init_args: AllowedInfoValueTypes | None,
    ) -> None:
        # Connect to the gRPC server with timeout
        self.channel = grpc.insecure_channel(server_address)
        try:
            # Wait for the channel to be ready
            grpc.channel_ready_future(self.channel).result(timeout=timeout)
        except grpc.FutureTimeoutError as err:
            self.channel.close()
            raise TimeoutError(
                f"Could not connect to server at {server_address} within {timeout} seconds"
            ) from err

        self.stub = AgentServiceStub(self.channel)

        # Initialize the remote environment
        init_request = AgentInitRequest()

        if init_args:
            init_request.init_args = msgpack.packb(init_args, use_bin_type=True)

        # Call the Init method and get space information
        agent_init_response = self.stub.Init(init_request)

        # Set up observation space
        space_dict = {}
        for name, proto_space in agent_init_response.observation_space.items():
            space_dict[name] = native_to_numpy_space(proto_space)
        self.observation_space = spaces.Dict(space_dict)

        # Set up action space
        self.action_space = native_to_numpy_space(agent_init_response.action_space)

        self.init_info: dict[str, AllowedInfoValueTypes] = msgpack.unpackb(
            agent_init_response.info, raw=False
        )

    def get_action(self, observation: dict[str, AllowedTypes]) -> AllowedTypes:
        """Get an action from the agent."""
        # Convert numpy arrays to lists for serialization
        serializable_observation: dict[str, AllowedSerializableTypes] = {}
        for key, value in observation.items():
            serializable_observation[key] = numpy_to_native(value)
        observation_request = ObservationRequest(
            observation=msgpack.packb(serializable_observation, use_bin_type=True)
        )

        # Call the GetAction method
        action_response = self.stub.GetAction(observation_request)

        # Deserialize the action
        action: AllowedSerializableTypes = msgpack.unpackb(
            action_response.action, raw=False
        )
        numpy_action = native_to_numpy(action, self.action_space)

        return numpy_action


def agent_check(
    server_address: str = "localhost:50051",
    num_steps: int = 5,
    **init_args: AllowedInfoValueTypes,
) -> None:
    """
    Run a simple test of the EnvironmentClient.

    Args:
        server_address: The address of the server (e.g., "localhost:50051")
        num_steps: Number of steps to run in the test
    """
    logger = logging.getLogger(__name__)
    try:
        # Create a remote agent
        agent = CRLAgentClient(
            server_address,
            timeout=60.0,
            **init_args,
        )

        # Run a few steps
        for _ in range(num_steps):
            obs = agent.observation_space.sample()
            action = agent.get_action(obs)
            if not agent.action_space.contains(action):
                logger.error(
                    f"Action {action} not in action space {agent.action_space}"
                )

            logger.info(f"Observation: {obs}")
            logger.info(f"Action: {action}")

        # Print success message if no errors occurred
        logger.info("\nSuccess! The agent client is working correctly.")

    except Exception as e:
        logger.info(f"\nError: {e}")
        logger.info("Failed to connect to or interact with the agent server.")
        raise
