"""gRPC server factory for Gymnasium environments."""

# gRPC Server Implementation
import logging
import traceback
from abc import abstractmethod
from concurrent import futures
from typing import Any, cast

import grpc
import msgpack
import numpy as np
from gymnasium import Env, spaces
from numpy.typing import NDArray

from ..proto_pb2 import (
    Empty,
    EnvInitRequest,
    EnvInitResponse,
    EnvironmentType,
    RenderResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)
from ..proto_pb2_grpc import (
    EnvironmentServiceServicer,
    add_EnvironmentServiceServicer_to_server,
)
from ..utils import (
    AllowedInfoValueTypes,
    AllowedSerializableTypes,
    AllowedSpaces,
    native_to_numpy_vec,
    numpy_to_native_space,
)


class CRLVecGymEnvironment(
    Env[
        dict[str, NDArray[np.floating | np.integer]],
        NDArray[np.floating | np.integer],
    ]
):
    """Abstract base class for Vectorized Environments."""

    num_envs: int
    init_info: dict[str, AllowedInfoValueTypes] | None

    @abstractmethod
    def reset(  # type: ignore
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[
        dict[str, NDArray[np.floating | np.integer]],
        list[dict[str, AllowedInfoValueTypes]],
    ]:
        """Reset the environment."""
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError


class VecEnvironmentServicer(
    EnvironmentServiceServicer,
):
    """gRPC servicer that wraps the GymEnvironment."""

    def __init__(
        self,
        environment_class: type[CRLVecGymEnvironment],
    ) -> None:
        self.env: CRLVecGymEnvironment | None = None
        self.logger = logging.getLogger("containerl.vec_environment_server")
        self.environment_class = environment_class
        self.environment_type: EnvironmentType = EnvironmentType.VECTORIZED
        self.num_envs: int = 1
        self.space_type_map: dict[str, AllowedSpaces] = {}

    def Init(  # noqa: N802 #  gRPC method names use UpperCamelCase
        self, request: EnvInitRequest, context: grpc.ServicerContext
    ) -> EnvInitResponse:
        """Initialize the environment and return space information."""
        try:
            # Prepare initialization arguments
            init_args: dict[str, AllowedInfoValueTypes] = {}
            if request.HasField("init_args"):
                init_args = msgpack.unpackb(request.init_args, raw=False)

            # Create the environment with all arguments
            self.env = self.environment_class(**init_args)

            if not hasattr(self.env, "num_envs"):
                raise Exception(
                    "Vectorized environment must have 'num_envs' attribute."
                )

            self.num_envs = self.env.num_envs

            # Create response with space information
            response = EnvInitResponse()

            # Handle observation space (Dict space)
            if not isinstance(self.env.observation_space, spaces.Dict):
                raise Exception("Observation space must be a Dict")

            for space_name, space in self.env.observation_space.spaces.items():
                self.space_type_map[space_name] = space
                space_proto = response.observation_space[space_name]
                numpy_to_native_space(space, space_proto)

            # Handle action space
            if isinstance(self.env.action_space, spaces.MultiBinary):
                if len(self.env.action_space.shape) != 1:
                    raise Exception(
                        "MultiBinary action space must be 1D, consider flattening it."
                    )
            numpy_to_native_space(self.env.action_space, response.action_space)
            response.num_envs = self.num_envs
            response.environment_type = self.environment_type

            if not hasattr(self.env, "render_mode"):
                raise Exception("Environments must have 'render_mode' attribute.")
            response.render_mode = (
                self.env.render_mode if self.env.render_mode is not None else "None"
            )

            info = msgpack.packb(
                self.env.init_info if hasattr(self.env, "init_info") else {},  # pyright: ignore
                use_bin_type=True,
            )
            response.info = info

            return response
        except Exception as e:
            stack_trace = traceback.format_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(
                f"Error initializing environment: {str(e)}\nStacktrace: {stack_trace}"
            )
            return EnvInitResponse()

    def Reset(  # noqa: N802 #  gRPC method names use UpperCamelCase
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

    def Step(self, request: StepRequest, context: grpc.ServicerContext) -> StepResponse:  # noqa: N802 #  gRPC method names use UpperCamelCase
        """Take a step in the environment."""
        try:
            if self.env is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("Environment not initialized. Call Init first.")
                return StepResponse()

            # Deserialize the action
            action = msgpack.unpackb(request.action, raw=False)
            action = native_to_numpy_vec(action, self.env.action_space, self.num_envs)

            # Take a step in the environment
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Convert numpy arrays to lists for serialization
            serializable_obs = self._get_serializable_observation(obs)

            serializable_reward = reward.tolist()
            serializable_terminated = terminated.tolist()
            serializable_truncated = truncated.tolist()

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

    def Render(self, request: Empty, context: grpc.ServicerContext) -> RenderResponse:  # noqa: N802 #  gRPC method names use UpperCamelCase
        """Render the environment."""
        try:
            if self.env is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("Environment not initialized. Call Init first.")
                return RenderResponse()

            # Get the render output directly
            render_output = cast(Any, self.env.render())

            # If it's a numpy array, directly serialize it
            if isinstance(render_output, np.ndarray) and render_output.ndim == 3:
                # Create a dict with array metadata and data for proper reconstruction
                array_data: dict[str, tuple[int, ...] | str | bytes] = {
                    "shape": render_output.shape,  # pyright: ignore[reportUnknownMemberType]
                    "dtype": str(render_output.dtype),  # pyright: ignore
                    "data": render_output.tobytes(),
                }
                render_data = msgpack.packb(array_data, use_bin_type=True)
                return RenderResponse(render_data=render_data)
            else:
                # For non-array outputs, return empty data
                self.logger.warning(
                    "Render output is not an image, i.e. 3D, np.int8 numpy array; returning empty render data."
                )
                return RenderResponse(render_data=b"")
        except Exception as e:
            stack_trace = traceback.format_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(
                f"Error rendering environment: {str(e)}\nStacktrace: {stack_trace}"
            )
            return RenderResponse()

    def Close(self, request: Empty, context: grpc.ServicerContext) -> Empty:  # noqa: N802 #  gRPC method names use UpperCamelCase
        """Close the environment."""
        try:
            if self.env is not None:
                self.env.close()  # type: ignore
                self.env = None
            return Empty()
        except Exception as e:
            stack_trace = traceback.format_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(
                f"Error closing environment: {str(e)}\nStacktrace: {stack_trace}"
            )
            return Empty()

    def _get_serializable_observation(
        self, observation: dict[str, NDArray[np.floating | np.integer]]
    ) -> dict[str, list[AllowedSerializableTypes]]:
        return {key: value.tolist() for key, value in observation.items()}


def create_vec_environment_server(
    environment_class: type[CRLVecGymEnvironment],
    port: int = 50051,
) -> None:
    """Start the gRPC server."""
    logger = logging.getLogger("containerl.environment_server")
    environment_server = VecEnvironmentServicer(environment_class)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_EnvironmentServiceServicer_to_server(environment_server, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Environment server started, listening on port {port}")
    server.wait_for_termination()
