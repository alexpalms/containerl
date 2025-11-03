"""Client for connecting to a remote environment via gRPC."""

import logging
from typing import Any, SupportsFloat

import grpc
import msgpack
import numpy as np
from gymnasium import Env, spaces
from gymnasium.core import ActType, ObsType, RenderFrame

# Add the interface directory to the path to import the generated gRPC code
from containerl.interface.proto_pb2 import (
    Empty,
    EnvironmentType,
    InitRequest,
    ResetRequest,
    StepRequest,
)
from containerl.interface.proto_pb2_grpc import EnvironmentServiceStub
from containerl.interface.utils import (
    native_to_numpy,
    native_to_numpy_space,
    native_to_numpy_vec,
    numpy_to_native,
)


class EnvironmentClient(Env):
    """
    A Gym environment that connects to a remote environment via gRPC.

    This class implements the Gym interface and forwards all calls to a remote
    environment server running the AnyLogic Stock Management environment.

    Args:
        server_address (str): The address of the gRPC server (e.g., "localhost:50051")
        render_mode (str, optional): The render mode to use. Defaults to None.
        timeout (float, optional): The timeout for connecting to the gRPC server. Defaults to 60.0 seconds.
        init_args (dict, optional): Additional arguments to pass to the environment initialization.
    """

    def __init__(
        self,
        server_address: str,
        timeout: float = 60.0,
        render_mode: str | None = None,
        **init_args,
    ) -> None:
        # Connect to the gRPC server with timeout
        self.channel = grpc.insecure_channel(server_address)
        try:
            # Wait for the channel to be ready
            grpc.channel_ready_future(self.channel).result(timeout=timeout)
        except grpc.FutureTimeoutError:
            self.channel.close()
            raise TimeoutError(
                f"Could not connect to server at {server_address} within {timeout} seconds"
            )

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
        space_dict = {}
        for name, proto_space in spaces_response.observation_space.items():
            space_dict[name] = native_to_numpy_space(proto_space)
        self.observation_space = spaces.Dict(space_dict)

        # Set up action space
        self.action_space = native_to_numpy_space(spaces_response.action_space)

        # Set up number of environments
        if spaces_response.environment_type == EnvironmentType.VECTORIZED:
            self.num_envs = spaces_response.num_envs
        else:
            self.num_envs = 1

        self.environment_type = spaces_response.environment_type

        # Store render mode
        self.render_mode = (
            spaces_response.render_mode
            if spaces_response.render_mode != "None"
            else None
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and return the initial observation."""
        reset_request = ResetRequest()

        if seed is not None:
            reset_request.seed = seed

        if options is not None:
            reset_request.options = msgpack.packb(options, use_bin_type=True)

        # Call the Reset method
        reset_response = self.stub.Reset(reset_request)

        # Deserialize the observation and info
        observation = msgpack.unpackb(reset_response.observation, raw=False)
        info = msgpack.unpackb(reset_response.info, raw=False)

        # Convert lists back to numpy arrays for the observation
        numpy_observation = self._get_numpy_observation(observation)

        return numpy_observation, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step in the environment."""
        # Convert NumPy arrays to lists for serialization
        if self.environment_type == EnvironmentType.VECTORIZED:
            native_action = action.tolist()
        else:
            native_action = numpy_to_native(action, self.action_space)

        # Serialize the action
        serialized_action = msgpack.packb(native_action, use_bin_type=True)

        # Create the request
        step_request = StepRequest(action=serialized_action)

        # Call the Step method
        step_response = self.stub.Step(step_request)

        # Deserialize the observation and info
        observation = msgpack.unpackb(step_response.observation, raw=False)
        reward = msgpack.unpackb(step_response.reward, raw=False)
        terminated = msgpack.unpackb(step_response.terminated, raw=False)
        truncated = msgpack.unpackb(step_response.truncated, raw=False)
        info = msgpack.unpackb(step_response.info, raw=False)

        # Convert lists back to numpy arrays for the observation
        numpy_observation = self._get_numpy_observation(observation)
        if self.environment_type == EnvironmentType.VECTORIZED:
            reward = np.array(reward, dtype=np.float32).reshape(self.num_envs)
            terminated = np.array(terminated, dtype=bool).reshape(self.num_envs)
            truncated = np.array(truncated, dtype=bool).reshape(self.num_envs)

        return (numpy_observation, reward, terminated, truncated, info)

    def render(self) -> RenderFrame | list[RenderFrame] | str | None:
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
        except Exception:
            # If it's not msgpack data, it might be plain text or other format
            # Just return it as a string
            return render_response.render_data.decode("utf-8")

    def close(self):
        """Close the environment."""
        # Create the request
        close_request = Empty()

        # Call the Close method
        self.stub.Close(close_request)

        # Close the gRPC channel
        self.channel.close()

    def _get_numpy_observation(self, observation):
        if self.environment_type == EnvironmentType.VECTORIZED:
            return {
                key: native_to_numpy_vec(
                    value, self.observation_space[key], self.num_envs
                )
                for key, value in observation.items()
            }
        else:
            return {
                key: native_to_numpy(value, self.observation_space[key])
                for key, value in observation.items()
            }


def main(server_address: str = "localhost:50051", num_steps: int = 5) -> None:
    """
    Run a simple test of the EnvironmentClient.

    Args:
        server_address: The address of the server (e.g., "localhost:50051")
        num_steps: Number of steps to run in the test
    """
    logger = logging.getLogger("containerl.environment_client")
    try:
        # Create a remote environment
        env = EnvironmentClient(server_address)

        # Reset the environment
        obs, info = env.reset()
        if env.environment_type == EnvironmentType.VECTORIZED:
            for key, value in obs.items():
                for i in range(env.num_envs):
                    if not env.observation_space[key].contains(value[i]):
                        logger.error(
                            f"Initial observation is not of type {env.observation_space[key]}"
                        )
        else:
            if not env.observation_space.contains(obs):
                logger.error(
                    f"Initial observation is not of type {env.observation_space}"
                )
        logger.info(f"Initial observation: {obs}")
        logger.info(f"Initial info: {info}")

        # Run a few steps
        for i in range(num_steps):
            if env.environment_type == EnvironmentType.VECTORIZED:
                action = np.stack(
                    [env.action_space.sample() for _ in range(env.num_envs)]
                )
            else:
                action = env.action_space.sample()  # Random action
            if env.render_mode == "rgb_array":
                frame = env.render()
                assert isinstance(frame, np.ndarray), (
                    "Render mode is rgb_array, but render returned a non-array"
                )
                assert frame.shape[2] == 3, (
                    "Render mode is rgb_array, but render returned an array with the wrong number of channels"
                )
                logger.info("Rendering works as expected")
            obs, reward, terminated, truncated, info = env.step(action)
            if env.environment_type == EnvironmentType.VECTORIZED:
                for key, value in obs.items():
                    for i in range(env.num_envs):
                        if not env.observation_space[key].contains(value[i]):
                            logger.error(
                                f"Observation is not of type {env.observation_space[key]}"
                            )
            else:
                if not env.observation_space.contains(obs):
                    logger.error(f"Observation is not of type {env.observation_space}")

            logger.info(f"Observation: {obs}")
            logger.info(f"Info: {info}")
            logger.info(f"Action: {action}")
            logger.info(f"Reward: {reward}")
            logger.info(f"Terminated: {terminated}")
            logger.info(f"Truncated: {truncated}")

            if env.environment_type == EnvironmentType.VECTORIZED:
                done = np.any(terminated) or np.any(truncated)
            else:
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
