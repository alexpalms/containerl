"""Client for connecting to a remote agent via gRPC."""

import logging

import grpc
import msgpack
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from ..proto_pb2 import Empty, ObservationRequest

# Add the interface directory to the path to import the generated gRPC code
from ..proto_pb2_grpc import AgentServiceStub
from ..utils import (
    native_to_numpy,
    native_to_numpy_space,
    numpy_to_native,
)


class AgentClient:
    """
    A Gym environment compatible agent that connects to a remote environment via gRPC.

    This class implements the Gym interface and forwards all calls to a remote
    agent server.
    """

    def __init__(self, server_address: str, timeout: float = 60.0):
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
        init_request = Empty()

        # Call the Init method and get space information
        spaces_response = self.stub.GetSpaces(init_request)

        # Set up observation space
        space_dict = {}
        for name, proto_space in spaces_response.observation_space.items():
            space_dict[name] = native_to_numpy_space(proto_space)
        self.observation_space = spaces.Dict(space_dict)

        # Set up action space
        self.action_space = native_to_numpy_space(spaces_response.action_space)

    def get_action(self, observation: ObsType) -> ActType:
        """Get an action from the agent."""
        # Convert numpy arrays to lists for serialization
        serializable_observation = {}
        for key, value in observation.items():
            serializable_observation[key] = numpy_to_native(
                value, self.observation_space[key]
            )
        observation_request = ObservationRequest(
            observation=msgpack.packb(serializable_observation, use_bin_type=True)
        )

        # Call the GetAction method
        action_response = self.stub.GetAction(observation_request)

        # Deserialize the action
        action = msgpack.unpackb(action_response.action, raw=False)
        numpy_action = native_to_numpy(action, self.action_space)

        return numpy_action

    def get_action_serve(self, observation: ObsType) -> ActType:
        """Get an action from the agent."""
        # Convert numpy arrays to lists for serialization
        observation_request = ObservationRequest(
            observation=msgpack.packb(observation, use_bin_type=True)
        )

        # Call the GetAction method
        action_response = self.stub.GetAction(observation_request)

        # Deserialize the action
        action = msgpack.unpackb(action_response.action, raw=False)
        return action


def main(server_address: str = "localhost:50051", num_steps: int = 5) -> None:
    """
    Run a simple test of the EnvironmentClient.

    Args:
        server_address: The address of the server (e.g., "localhost:50051")
        num_steps: Number of steps to run in the test
    """
    logger = logging.getLogger(__name__)
    try:
        # Create a remote agent
        agent = AgentClient(server_address)

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


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the AgentClient")
    parser.add_argument(
        "--address",
        default="localhost:50051",
        help="AServer address (default: localhost:50051)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of steps to run in the test (default: 5)",
    )

    args = parser.parse_args()

    try:
        main(args.address, args.steps)
    except Exception:
        import sys

        sys.exit(1)
