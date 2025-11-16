"""gRPC server factory for Agents."""

# gRPC Server Implementation
import logging
import traceback
from abc import abstractmethod
from collections.abc import Mapping
from concurrent import futures
from typing import Generic, cast, final

import grpc
import gymnasium as gym
import gymnasium.spaces as spaces
import msgpack

from ..proto_pb2 import (
    ActionResponse,
    Empty,
    ObservationRequest,
    SpacesResponse,
)
from ..proto_pb2_grpc import (
    AgentServiceServicer,
    add_AgentServiceServicer_to_server,
)
from ..utils import (
    AllowedTypes,
    CRLActType,
    CRLObsType,
    native_to_numpy,
    numpy_to_native,
    numpy_to_native_space,
)


class CRLAgent(Generic[CRLObsType, CRLActType]):
    """Abstract base class for agents."""

    observation_space: spaces.Space[CRLObsType]
    action_space: spaces.Space[CRLActType]

    @final
    def get_spaces(self) -> tuple[spaces.Space[CRLObsType], spaces.Space[CRLActType]]:
        """Return the observation and action spaces."""
        return self.observation_space, self.action_space

    @abstractmethod
    def get_action(self, observation: CRLObsType) -> CRLActType:
        """Given an observation, return an action."""
        pass


def build_agent_server(
    agent: CRLAgent[Mapping[str, AllowedTypes], AllowedTypes],
) -> AgentServiceServicer:
    """Create an AgentServicer class using the provided AgentClass."""

    class AgentServicer(AgentServiceServicer):
        """gRPC servicer that wraps the Agent."""

        def __init__(
            self, agent: CRLAgent[Mapping[str, AllowedTypes], AllowedTypes]
        ) -> None:
            self.agent = agent
            self.observation_space, self.action_space = self.agent.get_spaces()
            # Handle observation space (Dict space)
            if not isinstance(self.observation_space, gym.spaces.Dict):
                raise Exception("Observation space must be a Dict")

        def GetSpaces(
            self, request: Empty, context: grpc.ServicerContext
        ) -> SpacesResponse:
            """Return agent space information."""
            try:
                # Create response with space information
                response = SpacesResponse()

                for space_name, space in cast(
                    gym.spaces.Dict, self.observation_space
                ).spaces.items():
                    space_proto = response.observation_space[space_name]
                    numpy_to_native_space(space, space_proto)

                # Handle action space
                if isinstance(self.action_space, gym.spaces.MultiBinary):
                    if not len(self.action_space.shape) == 1:
                        raise Exception(
                            "MultiBinary action space must be 1D, consider flattening it."
                        )
                numpy_to_native_space(self.action_space, response.action_space)

                return response
            except Exception as e:
                stack_trace = traceback.format_exc()
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(
                    f"Error getting agent spaces: {str(e)}\nStacktrace: {stack_trace}"
                )
                return SpacesResponse()

        def GetAction(
            self, request: ObservationRequest, context: grpc.ServicerContext
        ) -> ActionResponse:
            """Get the action from the agent."""
            try:
                # Get the action from the agent
                # Convert lists back to numpy arrays for the observation
                observation = msgpack.unpackb(request.observation, raw=False)
                numpy_observation = {}
                for key, value in cast(
                    gym.spaces.Dict, self.observation_space
                ).spaces.items():
                    numpy_observation[key] = native_to_numpy(observation[key], value)
                action = self.agent.get_action(numpy_observation)

                # Convert numpy arrays to lists for serialization
                serializable_action = numpy_to_native(action, self.action_space)

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

    agent_server = AgentServicer(agent)
    return agent_server


def create_agent_server(
    agent: CRLAgent[Mapping[str, AllowedTypes], AllowedTypes], port: int = 50051
) -> None:
    """Start the gRPC server."""
    logger = logging.getLogger(__name__)
    agent_server = build_agent_server(agent)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_AgentServiceServicer_to_server(agent_server, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Agent server started, listening on port {port}")
    server.wait_for_termination()
