"""Client for connecting to a remote environment via gRPC."""

import logging
import sys
from typing import Any, SupportsFloat

import grpc
import gymnasium as gym
import msgpack
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

# Add the interface directory to the path to import the generated gRPC code
from ..proto_pb2 import (
    Empty,
    EnvironmentType,
    InitRequest,
    ResetRequest,
    StepRequest,
)
from ..proto_pb2_grpc import EnvironmentServiceStub
from ..utils import (
    AllowedInfoValueTypes,
    AllowedSerializableTypes,
    AllowedSpaces,
    AllowedTypes,
    native_to_numpy,
    native_to_numpy_space,
    numpy_to_native,
)


class CRLEnvironmentClient:
    """Client for connecting to a remote environment via gRPC."""

    def __init__(
        self,
        server_address: str,
        timeout: float = 60.0,
        render_mode: str | None = None,
        **init_args: dict[str, Any],
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

        self.stub = EnvironmentServiceStub(self.channel)

        # Initialize the remote environment
        init_request = InitRequest()
        if render_mode is not None:
            init_request.render_mode = render_mode

        if init_args:
            init_request.init_args = msgpack.packb(init_args, use_bin_type=True)

        # Call the Init method and get space information
        spaces_response = self.stub.Init(init_request)

        # Set up observation space
        space_dict: dict[str, AllowedSpaces] = {}
        for name, proto_space in spaces_response.observation_space.items():
            space_dict[name] = native_to_numpy_space(proto_space)
        self.observation_space = spaces.Dict(space_dict)

        # Set up action space
        self.action_space = native_to_numpy_space(spaces_response.action_space)

        # Set up number of environments
        self.num_envs = spaces_response.num_envs
        self.environment_type = spaces_response.environment_type
        if self.environment_type != EnvironmentType.STANDARD:
            raise Exception(
                "EnvironmentClient only supports STANDARD environments. "
                "For VECTORIZED environments, please use EnvironmentClientVec."
            )

        # Store render mode
        self.render_mode = (
            spaces_response.render_mode
            if spaces_response.render_mode != "None"
            else None
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[
        dict[str, AllowedTypes],
        dict[str, AllowedInfoValueTypes],
    ]:
        """Reset the environment and return the initial observation."""
        reset_request = ResetRequest()

        if seed is not None:
            reset_request.seed = seed

        if options is not None:
            reset_request.options = msgpack.packb(options, use_bin_type=True)

        # Call the Reset method
        reset_response = self.stub.Reset(reset_request)

        # Deserialize the observation and info
        observation: dict[str, AllowedSerializableTypes] = msgpack.unpackb(
            reset_response.observation, raw=False
        )
        info: dict[str, AllowedInfoValueTypes] = msgpack.unpackb(
            reset_response.info, raw=False
        )

        # Convert lists back to numpy arrays for the observation
        numpy_observation = self._get_numpy_observation(observation)

        return numpy_observation, info

    def step(
        self, action: AllowedTypes
    ) -> tuple[
        dict[str, AllowedTypes],
        SupportsFloat,
        bool,
        bool,
        dict[str, AllowedInfoValueTypes],
    ]:
        """Take a step in the environment."""
        # Convert NumPy arrays to lists for serialization
        if self.environment_type == EnvironmentType.VECTORIZED:
            native_action = action.tolist()
        else:
            native_action = numpy_to_native(action)

        # Serialize the action
        serialized_action = msgpack.packb(native_action, use_bin_type=True)

        # Create the request
        step_request = StepRequest(action=serialized_action)

        # Call the Step method
        step_response = self.stub.Step(step_request)

        # Deserialize the observation and info
        observation: dict[str, AllowedSerializableTypes] = msgpack.unpackb(
            step_response.observation, raw=False
        )
        reward: SupportsFloat = msgpack.unpackb(step_response.reward, raw=False)
        terminated: bool = msgpack.unpackb(step_response.terminated, raw=False)
        truncated: bool = msgpack.unpackb(step_response.truncated, raw=False)
        info: dict[str, AllowedInfoValueTypes] = msgpack.unpackb(
            step_response.info, raw=False
        )

        # Convert lists back to numpy arrays for the observation
        numpy_observation = self._get_numpy_observation(observation)

        return (numpy_observation, reward, terminated, truncated, info)

    def render(self) -> NDArray[np.uint8] | None:
        """Render the environment."""
        # Create the request
        render_request = Empty()

        # Call the Render method
        render_response = self.stub.Render(render_request)

        # If no render data was returned, return None
        if not render_response.render_data:
            return None

        try:
            # Try to deserialize as a msgpack object (for numpy arrays)
            render_data = msgpack.unpackb(render_response.render_data, raw=False)

            array = np.frombuffer(
                render_data["data"], dtype=np.dtype(render_data["dtype"])
            )
            array = array.reshape(render_data["shape"])
            return array
        except Exception as err:
            raise Exception("Failed to deserialize render data from server") from err

    def close(self) -> None:
        """Close the environment."""
        # Create the request
        close_request = Empty()

        # Call the Close method
        self.stub.Close(close_request)

        # Close the gRPC channel
        self.channel.close()

    def _get_numpy_observation(
        self, observation: dict[str, AllowedSerializableTypes]
    ) -> dict[str, AllowedTypes]:
        return {
            key: native_to_numpy(value, self.observation_space[key])
            for key, value in observation.items()
        }


class CRLGymEnvironmentAdapter(gym.Env[dict[str, AllowedTypes], AllowedTypes]):
    """Adapter to use CRLEnvironmentClient as a Gym environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, server_address: str, timeout: float = 60.0, **init_args: Any):
        self.client = CRLEnvironmentClient(server_address, timeout=timeout, **init_args)
        self.observation_space = self.client.observation_space
        self.action_space = self.client.action_space
        self.render_mode = self.client.render_mode

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, AllowedTypes], dict[str, AllowedInfoValueTypes]]:
        """Reset the environment."""
        return self.client.reset(seed=seed, options=options)

    def step(
        self, action: AllowedTypes
    ) -> tuple[
        dict[str, AllowedTypes],
        SupportsFloat,
        bool,
        bool,
        dict[str, AllowedInfoValueTypes],
    ]:
        """Take a step in the environment."""
        return self.client.step(action)

    def render(self) -> NDArray[np.uint8] | None:
        """Render the environment."""
        return self.client.render()

    def close(self) -> None:
        """Close the environment."""
        self.client.close()


def main(server_address: str = "localhost:50051", num_steps: int = 5) -> None:
    """
    Run a simple test of the EnvironmentClient.

    Args:
        server_address: The address of the server (e.g., "localhost:50051")
        num_steps: Number of steps to run in the test
    """
    logger = logging.getLogger("containerl.environment_client")
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    )
    try:
        # Create a remote environment
        env = CRLEnvironmentClient(server_address)

        # Reset the environment
        obs, info = env.reset()
        if not env.observation_space.contains(obs):
            logger.error(f"Initial observation is not of type {env.observation_space}")
        logger.info(f"Initial observation: {obs}")
        logger.info(f"Initial info: {info}")

        # Run a few steps
        for _ in range(num_steps):
            action = env.action_space.sample()  # Random action
            if env.render_mode == "rgb_array":
                frame = env.render()
                if not isinstance(frame, np.ndarray):
                    logger.error(
                        f"Render mode is rgb_array, but render returned a non-array, type: {type(frame)}"
                    )
                    raise ValueError("Render returned non-array in rgb_array mode")
                if frame.shape[2] != 3:
                    logger.error(
                        "Render mode is rgb_array, but render returned an array with the wrong number of channels"
                    )
                    raise ValueError(
                        "Render returned an array with the wrong number of channels"
                    )
                logger.info("Rendering works as expected")
            obs, reward, terminated, truncated, info = env.step(action)
            if not env.observation_space.contains(obs):
                logger.error(f"Observation is not of type {env.observation_space}")

            logger.info(f"Observation: {obs}")
            logger.info(f"Info: {info}")
            logger.info(f"Action: {action}")
            logger.info(f"Reward: {reward}")
            logger.info(f"Terminated: {terminated}")
            logger.info(f"Truncated: {truncated}")

            done = terminated or truncated
            if done:
                obs, info = env.reset()

        # Close the environment
        env.close()

        # Print success message if no errors occurred
        logger.info("\nSuccess! The environment client is working correctly.")

    except Exception as e:
        logger.info(f"\nError: {e}")
        logger.info("Failed to connect to or interact with the environment server.")
        raise


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the EnvironmentClient")
    parser.add_argument(
        "--address",
        default="localhost:50051",
        help="Server address (default: localhost:50051)",
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
