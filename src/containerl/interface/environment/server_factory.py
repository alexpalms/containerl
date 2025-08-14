# gRPC Server Implementation
import msgpack
import numpy as np
import grpc
import traceback
import gymnasium as gym
from concurrent import futures
from containerl.interface import add_EnvironmentServiceServicer_to_server
from containerl.interface import EnvironmentType
from containerl.interface import EnvironmentService, SpacesResponse, ResetResponse, StepResponse, RenderResponse, Empty
from containerl.interface.utils import numpy_to_native, native_to_numpy, numpy_to_native_space, native_to_numpy_vec


class EnvironmentServicer(EnvironmentService):
    """gRPC servicer that wraps the GymEnvironment."""

    def __init__(self, environment_class):
        self.env = None
        self.environment_class = environment_class
        self.environment_type = None
        self.num_envs = None

    def Init(self, request, context):
        """Initialize the environment and return space information."""
        try:
            # Prepare initialization arguments
            init_args = {}
            if request.HasField('init_args'):
                init_args = msgpack.unpackb(request.init_args, raw=False)

            # Add render_mode to init_args if provided
            if request.HasField('render_mode'):
                init_args['render_mode'] = request.render_mode

            # Create the environment with all arguments
            self.env = self.environment_class(**init_args)

            # Create response with space information
            response = SpacesResponse()

            # Handle observation space (Dict space)
            assert isinstance(self.env.observation_space, gym.spaces.Dict), "Observation space must be a Dict"
            for space_name, space in self.env.observation_space.spaces.items():
                space_proto = response.observation_space[space_name]
                numpy_to_native_space(space, space_proto)

            # Handle action space
            action_space = self.env.action_space
            if isinstance(action_space, gym.spaces.MultiBinary):
                assert len(action_space.shape) == 1, "MultiBinary action space must be 1D, consider flattening it."
            numpy_to_native_space(action_space, response.action_space)

            if hasattr(self.env, 'num_envs'):
                self.num_envs = self.env.num_envs
                self.environment_type = EnvironmentType.VECTORIZED
            else:
                self.environment_type = EnvironmentType.STANDARD
                self.num_envs = 1

            response.num_envs = self.num_envs
            response.environment_type = self.environment_type
            response.render_mode = self.env.render_mode if self.env.render_mode is not None else "None"

            return response
        except Exception as e:
            stack_trace = traceback.format_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error initializing environment: {str(e)}\nStacktrace: {stack_trace}")
            return SpacesResponse()

    def Reset(self, request, context):
        """Reset the environment and return the initial observation."""
        try:
            if self.env is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("Environment not initialized. Call Init first.")
                return ResetResponse()

            # Extract seed and options if provided
            seed = None
            if request.HasField('seed'):
                seed = request.seed

            options = None
            if request.HasField('options'):
                options = msgpack.unpackb(request.options, raw=False)

            # Reset the environment
            obs, info = self.env.reset(seed=seed, options=options)

            # Convert numpy arrays to lists for serialization
            serializable_observation = self._get_serializable_observation(obs)

            # Serialize the observation and info
            response = ResetResponse(
                observation=msgpack.packb(serializable_observation, use_bin_type=True),
                info=msgpack.packb(info, use_bin_type=True)
            )

            return response
        except Exception as e:
            stack_trace = traceback.format_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error resetting environment: {str(e)}\nStacktrace: {stack_trace}")
            return ResetResponse()

    def Step(self, request, context):
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
                action = native_to_numpy_vec(action, self.env.action_space, self.num_envs)
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
                info=msgpack.packb(info, use_bin_type=True)
            )

            return response
        except Exception as e:
            stack_trace = traceback.format_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error during environment step: {str(e)}\nStacktrace: {stack_trace}")
            return StepResponse()

    def Render(self, request, context):
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
                    'shape': render_output.shape,
                    'dtype': str(render_output.dtype),
                    'data': render_output.tobytes(),
                }
                render_data = msgpack.packb(array_data, use_bin_type=True)
                return RenderResponse(render_data=render_data)
            else:
                # For non-array outputs, return empty data
                return RenderResponse(render_data=b'')
        except Exception as e:
            stack_trace = traceback.format_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error rendering environment: {str(e)}\nStacktrace: {stack_trace}")
            return RenderResponse()

    def Close(self, request, context):
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
            context.set_details(f"Error closing environment: {str(e)}\nStacktrace: {stack_trace}")
            return Empty()

    def _get_serializable_observation(self, observation):
        if self.environment_type == EnvironmentType.VECTORIZED:
            return {key: value.tolist() for key, value in observation.items()}
        else:
            return {key: numpy_to_native(value, self.env.observation_space[key]) for key, value in observation.items()}

def create_environment_server(environment_class, port=50051):
    """Start the gRPC server."""
    environment_server = EnvironmentServicer(environment_class)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_EnvironmentServiceServicer_to_server(environment_server, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Environment server started, listening on port {port}")
    server.wait_for_termination()