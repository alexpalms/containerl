"""gRPC server factory for Agents."""

# gRPC Server Implementation
import logging
import traceback
from concurrent import futures

import grpc
import gymnasium as gym
import msgpack

from containerl.interface.proto_pb2 import ActionResponse, SpacesResponse
from containerl.interface.proto_pb2_grpc import (
    AgentService,
    add_AgentServiceServicer_to_server,
)
from containerl.interface.utils import (
    native_to_numpy,
    numpy_to_native,
    numpy_to_native_space,
)


def build_agent_server(agent) -> AgentService:
    """
    Factory function that creates an AgentServicer class using the provided AgentClass.

    Args:
        AgentClass: The agent class to use for creating agents

    Returns
    -------
        A configured AgentServicer class
    """

    class AgentServicer(AgentService):
        """gRPC servicer that wraps the Agent."""

        def __init__(self):
            self.agent = agent

        def GetSpaces(self, request, context):
            """Return agent space information."""
            try:
                # Create response with space information
                response = SpacesResponse()

                # Handle observation space (Dict space)
                assert isinstance(self.agent.observation_space, gym.spaces.Dict), (
                    "Observation space must be a Dict"
                )
                for space_name, space in self.agent.observation_space.spaces.items():
                    space_proto = response.observation_space[space_name]
                    numpy_to_native_space(space, space_proto)

                # Handle action space
                action_space = self.agent.action_space
                if isinstance(action_space, gym.spaces.MultiBinary):
                    assert len(action_space.shape) == 1, (
                        "MultiBinary action space must be 1D, consider flattening it."
                    )
                numpy_to_native_space(action_space, response.action_space)

                return response
            except Exception as e:
                stack_trace = traceback.format_exc()
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(
                    f"Error getting agent spaces: {str(e)}\nStacktrace: {stack_trace}"
                )
                return SpacesResponse()

        def GetAction(self, request, context):
            """Get the action from the agent."""
            try:
                if self.agent is None:
                    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                    context.set_details("Agent not initialized.")
                    return ActionResponse()

                # Get the action from the agent
                # Convert lists back to numpy arrays for the observation
                observation = msgpack.unpackb(request.observation, raw=False)
                numpy_observation = {}
                for key, value in observation.items():
                    numpy_observation[key] = native_to_numpy(
                        value, self.agent.observation_space[key]
                    )
                action = self.agent.get_action(numpy_observation)

                # Convert numpy arrays to lists for serialization
                serializable_action = numpy_to_native(action, self.agent.action_space)

                # Serialize the observation and info
                response = ActionResponse(
                    action=msgpack.packb(serializable_action, use_bin_type=True)
                )

                return response
            except Exception as e:
                stack_trace = traceback.format_exc()
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(
                    f"Error getting agent action: {str(e)}\nStacktrace: {stack_trace}"
                )
                return ActionResponse()

    agent_server = AgentServicer()
    return agent_server


def create_agent_server(agent, port=50051) -> None:
    """Start the gRPC server."""
    logger = logging.getLogger(__name__)
    agent_server = build_agent_server(agent)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_AgentServiceServicer_to_server(agent_server, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Agent server started, listening on port {port}")
    server.wait_for_termination()
