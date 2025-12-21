"""Client for connecting to a remote environment via gRPC."""

import logging
import sys
from typing import cast

import grpc
import msgpack
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

# Add the interface directory to the path to import the generated gRPC code
from ..proto_pb2 import (
    Empty,
    EnvInitRequest,
    EnvironmentType,
    ResetRequest,
    StepRequest,
)
from ..proto_pb2_grpc import EnvironmentServiceStub
from ..utils import (
    AllowedInfoValueTypes,
    AllowedSpaces,
    native_to_numpy_space,
    native_to_numpy_vec,
)
from .server_factory_vec import CRLVecGymEnvironment


class CRLVecEnvironmentClient:
    """A Gym environment that connects to a remote environment via gRPC."""

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

        self.stub = EnvironmentServiceStub(self.channel)

        # Initialize the remote environment
        init_request = EnvInitRequest()

        if init_args:
            init_request.init_args = msgpack.packb(init_args, use_bin_type=True)

        # Call the Init method and get space information
        env_init_response = self.stub.Init(init_request)

        # Set up observation space
        space_dict: dict[str, AllowedSpaces] = {}
        for name, proto_space in env_init_response.observation_space.items():
            space_dict[name] = native_to_numpy_space(proto_space)
        self.observation_space = spaces.Dict(space_dict)

        # Set up action space
        self.action_space = cast(
            spaces.Space[NDArray[np.floating | np.integer]],
            native_to_numpy_space(env_init_response.action_space),
        )

        # Set up number of environments
        self.num_envs = env_init_response.num_envs
        self.environment_type = env_init_response.environment_type
        if self.environment_type != EnvironmentType.VECTORIZED:
            raise Exception(
                "VecEnvironmentClient only supports VECTORIZED environments. "
                "For STANDARD environments, please use EnvironmentClient."
            )

        # Store render mode
        self.render_mode = (
            env_init_response.render_mode
            if env_init_response.render_mode != "None"
            else None
        )

        # Store Environment intialization info if provided
        self.init_info: dict[str, AllowedInfoValueTypes] = msgpack.unpackb(
            env_init_response.info, raw=False
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, AllowedInfoValueTypes] | None = None,
    ) -> tuple[
        dict[str, NDArray[np.floating | np.integer]],
        list[dict[str, AllowedInfoValueTypes]],
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
        observation: dict[str, list[float | int]] = msgpack.unpackb(
            reset_response.observation, raw=False
        )
        info: list[dict[str, AllowedInfoValueTypes]] = msgpack.unpackb(
            reset_response.info, raw=False
        )

        # Convert lists back to numpy arrays for the observation
        numpy_observation = self._get_numpy_observation(observation)

        return numpy_observation, info

    def step(
        self, action: NDArray[np.floating | np.integer]
    ) -> tuple[
        dict[str, NDArray[np.floating | np.integer]],
        NDArray[np.floating],
        NDArray[np.bool_],
        NDArray[np.bool_],
        list[dict[str, AllowedInfoValueTypes]],
    ]:
        """Take a step in the environment."""
        # Convert NumPy arrays to lists for serialization
        native_action = action.tolist()

        # Serialize the action
        serialized_action = msgpack.packb(native_action, use_bin_type=True)

        # Create the request
        step_request = StepRequest(action=serialized_action)

        # Call the Step method
        step_response = self.stub.Step(step_request)

        # Deserialize the observation and info
        observation: dict[str, list[float | int]] = msgpack.unpackb(
            step_response.observation, raw=False
        )
        reward: list[float] = msgpack.unpackb(step_response.reward, raw=False)
        terminated: list[bool] = msgpack.unpackb(step_response.terminated, raw=False)
        truncated: list[bool] = msgpack.unpackb(step_response.truncated, raw=False)
        info: list[dict[str, AllowedInfoValueTypes]] = msgpack.unpackb(
            step_response.info, raw=False
        )

        # Convert lists back to numpy arrays for the observation
        numpy_observation = self._get_numpy_observation(observation)
        numpy_reward = np.array(reward, dtype=np.float32).reshape(self.num_envs)
        numpy_terminated = np.array(terminated, dtype=bool).reshape(self.num_envs)
        numpy_truncated = np.array(truncated, dtype=bool).reshape(self.num_envs)

        return (
            numpy_observation,
            numpy_reward,
            numpy_terminated,
            numpy_truncated,
            info,
        )

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
        self, observation: dict[str, list[float | int]]
    ) -> dict[str, NDArray[np.floating | np.integer]]:
        return {
            key: native_to_numpy_vec(value, self.observation_space[key], self.num_envs)
            for key, value in observation.items()
        }


class CRLVecGymEnvironmentAdapter(CRLVecGymEnvironment):
    """Adapter to use CRLEnvironmentClient as a Gym environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        server_address: str,
        timeout: float = 60.0,
        **init_args: AllowedInfoValueTypes | None,
    ):
        self.client = CRLVecEnvironmentClient(
            server_address, timeout=timeout, **init_args
        )
        self.observation_space = self.client.observation_space
        self.action_space = self.client.action_space
        self.render_mode = self.client.render_mode
        self.init_info = self.client.init_info
        self.num_envs = self.client.num_envs
        self.environment_type = self.client.environment_type

    def reset(  # type: ignore
        self,
        *,
        seed: int | None = None,
        options: dict[str, AllowedInfoValueTypes] | None = None,
    ) -> tuple[
        dict[str, NDArray[np.floating | np.integer]],
        list[dict[str, AllowedInfoValueTypes]],
    ]:
        """Reset the environment."""
        return self.client.reset(seed=seed, options=options)

    def step(  # type: ignore
        self, action: NDArray[np.floating | np.integer]
    ) -> tuple[
        dict[str, NDArray[np.floating | np.integer]],
        NDArray[np.floating],
        NDArray[np.bool_],
        NDArray[np.bool_],
        list[dict[str, AllowedInfoValueTypes]],
    ]:
        """Take a step in the environment."""
        return self.client.step(action)

    def render(self) -> NDArray[np.uint8] | None:  # type: ignore
        """Render the environment."""
        return self.client.render()

    def close(self) -> None:
        """Close the environment."""
        self.client.close()


def vec_environment_check(
    server_address: str = "localhost:50051", num_steps: int = 5
) -> None:
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
        env = CRLVecEnvironmentClient(server_address)
        # Reset the environment
        obs, info = env.reset()
        for key, value in obs.items():
            for i in range(env.num_envs):
                if not env.observation_space[key].contains(value[i]):
                    logger.error(
                        f"Initial observation is not of type {env.observation_space[key]}"
                    )

        logger.info(f"Initial observation: {obs}")
        logger.info(f"Initial info: {info}")

        # Run a few steps
        for i in range(num_steps):
            action = np.stack([env.action_space.sample() for _ in range(env.num_envs)])

            if env.render_mode == "rgb_array":
                frame = env.render()
                if not isinstance(frame, np.ndarray):
                    logger.error(
                        f"Render did not return a numpy array, type: {type(frame)}"
                    )
                    raise Exception(
                        "Render mode is rgb_array, but render returned a non-array"
                    )
                if not frame.shape[2] == 3:
                    logger.error(
                        "Render returned an array with the wrong number of channels"
                    )
                    raise Exception(
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

            if np.any(terminated) or np.any(truncated):
                obs, info = env.reset()

        # Close the environment
        env.close()

        # Print success message if no errors occurred
        logger.info("\nSuccess! The environment client is working correctly.")

    except Exception as e:
        logger.info(f"\nError: {e}")
        logger.info("Failed to connect to or interact with the environment server.")
        raise
