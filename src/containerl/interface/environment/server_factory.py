"""gRPC server factory for Gymnasium environments."""

# gRPC Server Implementation
import logging
import traceback
from concurrent import futures
from typing import Generic

import grpc
import gymnasium as gym
import msgpack
import numpy as np

from containerl import (
    AllowedSpaces,
    CRLActType,
    CRLObsType,
    native_to_numpy,
    native_to_numpy_vec,
    numpy_to_native,
    numpy_to_native_space,
)
from containerl.interface.proto_pb2 import (
    Empty,
    EnvironmentType,
    InitRequest,
    RenderResponse,
    ResetRequest,
    ResetResponse,
    SpacesResponse,
    StepRequest,
    StepResponse,
)
from containerl.interface.proto_pb2_grpc import (
    EnvironmentServiceServicer,
    add_EnvironmentServiceServicer_to_server,
)


class CRLEnvironment(gym.Env[CRLObsType, CRLActType]):
    """Abstract base class for Environments."""


class EnvironmentServicer(Generic[CRLObsType, CRLActType], EnvironmentServiceServicer):
    """gRPC servicer that wraps the GymEnvironment."""

    def __init__(
        self,
        environment_class: type[CRLEnvironment[CRLObsType, CRLActType]],
    ) -> None:
        self.env: CRLEnvironment[CRLObsType, CRLActType] | None = None
        self.environment_class = environment_class
        self.environment_type: EnvironmentType | None = None
        self.num_envs: int | None = None
        self.space_type_map: dict[str, AllowedSpaces] = {}

    def Init(
        self, request: InitRequest, context: grpc.ServicerContext
    ) -> SpacesResponse:
        """Initialize the environment and return space information."""
        try:
            # Prepare initialization arguments
            init_args = {}
            if request.HasField("init_args"):
                init_args = msgpack.unpackb(request.init_args, raw=False)

            # Add render_mode to init_args if provided
            if request.HasField("render_mode"):
                init_args["render_mode"] = request.render_mode

            # Create the environment with all arguments
            self.env = self.environment_class(**init_args)

            # Create response with space information
            response = SpacesResponse()

            # Handle observation space (Dict space)
            if not isinstance(self.env.observation_space, gym.spaces.Dict):
                raise Exception("Observation space must be a Dict")

            for space_name, space in self.env.observation_space.spaces.items():
                self.space_type_map[space_name] = space
                space_proto = response.observation_space[space_name]
                numpy_to_native_space(space, space_proto)

            # Handle action space
            action_space = self.env.action_space
            if isinstance(action_space, gym.spaces.MultiBinary):
                if len(action_space.shape) != 1:
                    raise Exception(
                        "MultiBinary action space must be 1D, consider flattening it."
                    )
            numpy_to_native_space(action_space, response.action_space)

            if hasattr(self.env, "num_envs"):
                self.num_envs = self.env.num_envs
                self.environment_type = EnvironmentType.VECTORIZED
            else:
                self.environment_type = EnvironmentType.STANDARD
                self.num_envs = 1

            response.num_envs = self.num_envs
            response.environment_type = self.environment_type
            response.render_mode = (
                self.env.render_mode if self.env.render_mode is not None else "None"
            )

            return response
        except Exception as e:
            stack_trace = traceback.format_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(
                f"Error initializing environment: {str(e)}\nStacktrace: {stack_trace}"
            )
            return SpacesResponse()

    def Reset(
        self,
        request: ResetRequest,
        context: grpc.ServicerContext,
    ) -> ResetResponse:
        """Reset the environment and return the initial observation."""
        try:
            if self.env is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("Environment not initialized. Call Init first.")
                return ResetResponse()

            # Extract seed and options if provided
            seed = None
            if request.HasField("seed"):
                seed = request.seed

            options = None
            if request.HasField("options"):
                options = msgpack.unpackb(request.options, raw=False)

            # Reset the environment
            obs, info = self.env.reset(seed=seed, options=options)

            # Convert numpy arrays to lists for serialization
            serializable_observation = self._get_serializable_observation(obs)

            # Serialize the observation and info
            response = ResetResponse(
                observation=msgpack.packb(serializable_observation, use_bin_type=True),
                info=msgpack.packb(info, use_bin_type=True),
            )

            return response
        except Exception as e:
            stack_trace = traceback.format_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(
                f"Error resetting environment: {str(e)}\nStacktrace: {stack_trace}"
            )
            return ResetResponse()

    def Step(self, request: StepRequest, context: grpc.ServicerContext) -> StepResponse:
        """Take a step in the environment."""
        try:
            if self.env is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("Environment not initialized. Call Init first.")
                return StepResponse()

            # Deserialize the action
            action = msgpack.unpackb(request.action, raw=False)

            # Convert lists back to numpy
            if self.environment_type == EnvironmentType.VECTORIZED:
                action = native_to_numpy_vec(
                    action, self.env.action_space, self.num_envs
                )
            else:
                action = native_to_numpy(action, self.env.action_space)

            # Take a step in the environment
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Convert numpy arrays to lists for serialization
            serializable_obs = self._get_serializable_observation(obs)

            # Create and return the response
            if self.environment_type == EnvironmentType.VECTORIZED:
                serializable_reward = reward.tolist()
                serializable_terminated = terminated.tolist()
                serializable_truncated = truncated.tolist()
            else:
                serializable_reward = float(reward)
                serializable_terminated = bool(terminated)
                serializable_truncated = bool(truncated)

            response = StepResponse(
                observation=msgpack.packb(serializable_obs, use_bin_type=True),
                reward=msgpack.packb(serializable_reward, use_bin_type=True),
                terminated=msgpack.packb(serializable_terminated, use_bin_type=True),
                truncated=msgpack.packb(serializable_truncated, use_bin_type=True),
                info=msgpack.packb(info, use_bin_type=True),
            )

            return response
        except Exception as e:
            stack_trace = traceback.format_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(
                f"Error during environment step: {str(e)}\nStacktrace: {stack_trace}"
            )
            return StepResponse()

    def Render(self, request: Empty, context: grpc.ServicerContext) -> RenderResponse:
        """Render the environment."""
        try:
            if self.env is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("Environment not initialized. Call Init first.")
                return RenderResponse()

            # Get the render output directly
            render_output = self.env.render()

            # If it's a numpy array, directly serialize it
            if isinstance(render_output, np.ndarray):
                # Create a dict with array metadata and data for proper reconstruction
                array_data = {
                    "shape": render_output.shape,
                    "dtype": str(render_output.dtype),
                    "data": render_output.tobytes(),
                }
                render_data = msgpack.packb(array_data, use_bin_type=True)
                return RenderResponse(render_data=render_data)
            else:
                # For non-array outputs, return empty data
                return RenderResponse(render_data=b"")
        except Exception as e:
            stack_trace = traceback.format_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(
                f"Error rendering environment: {str(e)}\nStacktrace: {stack_trace}"
            )
            return RenderResponse()

    def Close(self, request: Empty, context: grpc.ServicerContext) -> Empty:
        """Close the environment."""
        try:
            if self.env is not None:
                self.env.close()
                self.env = None
                self.num_envs = None
                self.environment_type = None
            return Empty()
        except Exception as e:
            stack_trace = traceback.format_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(
                f"Error closing environment: {str(e)}\nStacktrace: {stack_trace}"
            )
            return Empty()

    def _get_serializable_observation(
        self, observation: dict[str, AllowedTypes]
    ) -> dict[str, list[int | float] | int]:
        if self.environment_type == EnvironmentType.VECTORIZED:
            return {key: value.tolist() for key, value in observation.items()}
        else:
            return {
                key: numpy_to_native(observation[key], space)
                for key, space in self.space_type_map.items()
            }


def create_environment_server(
    environment_class: type[CRLEnvironment[CRLObsType, CRLActType]],
    port: int = 50051,
) -> None:
    """Start the gRPC server."""
    logger = logging.getLogger("containerl.environment_server")
    environment_server = EnvironmentServicer(environment_class)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_EnvironmentServiceServicer_to_server(environment_server, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Environment server started, listening on port {port}")
    server.wait_for_termination()
