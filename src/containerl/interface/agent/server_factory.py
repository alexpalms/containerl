"""gRPC server factory for Agents."""

# gRPC Server Implementation
import logging
import traceback
from abc import ABC, abstractmethod
from concurrent import futures
from typing import cast

import grpc
import gymnasium as gym
import gymnasium.spaces as spaces
import msgpack

from ..proto_pb2 import (
    ActionResponse,
    AgentInitRequest,
    AgentInitResponse,
    ObservationRequest,
)
from ..proto_pb2_grpc import (
    AgentServiceServicer,
    add_AgentServiceServicer_to_server,
)
from ..utils import (
    AllowedInfoValueTypes,
    AllowedSerializableTypes,
    AllowedTypes,
    native_to_numpy,
    numpy_to_native,
    numpy_to_native_space,
)


class CRLAgent(ABC):
    """Abstract base class for agents."""

    observation_space: spaces.Space[dict[str, AllowedTypes]]
    action_space: spaces.Space[AllowedTypes]
    init_info: dict[str, AllowedInfoValueTypes]

    @abstractmethod
    def get_action(self, observation: dict[str, AllowedTypes]) -> AllowedTypes:
        """Given an observation, return an action."""
        pass


class AgentServicer(AgentServiceServicer):
    """gRPC servicer that wraps the Agent."""

    def __init__(self, agent_class: type[CRLAgent]) -> None:
        self.logger = logging.getLogger("containerl.agent_server")
        self.agent_class = agent_class
        self.agent: CRLAgent | None = None

    def Init(  # noqa: N802 #  gRPC method names use UpperCamelCase
        self, request: AgentInitRequest, context: grpc.ServicerContext
    ) -> AgentInitResponse:
        """Return agent space information."""
        try:
            # Prepare initialization arguments
            init_args: dict[str, AllowedInfoValueTypes] = {}
            if request.HasField("init_args"):
                init_args = msgpack.unpackb(request.init_args, raw=False)

            self.agent = self.agent_class(**init_args)
            self.observation_space = self.agent.observation_space
            action_space = self.agent.action_space

            # Handle observation space (Dict space)
            if not isinstance(self.observation_space, gym.spaces.Dict):
                raise Exception("Observation space must be a Dict")

            # Create response with space information
            response = AgentInitResponse()

            for space_name, space in self.observation_space.spaces.items():
                space_proto = response.observation_space[space_name]
                numpy_to_native_space(space, space_proto)

            # Handle action space
            if isinstance(action_space, gym.spaces.MultiBinary):
                if not len(action_space.shape) == 1:
                    raise Exception(
                        "MultiBinary action space must be 1D, consider flattening it."
                    )
            numpy_to_native_space(action_space, response.action_space)

            info = msgpack.packb(
                self.agent.init_info if hasattr(self.agent, "init_info") else {},
                use_bin_type=True,
            )
            response.info = info

            return response
        except Exception as e:
            stack_trace = traceback.format_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(
                f"Error getting agent spaces: {str(e)}\nStacktrace: {stack_trace}"
            )
            return AgentInitResponse()

    def GetAction(  # noqa: N802 #  gRPC method names use UpperCamelCase
        self, request: ObservationRequest, context: grpc.ServicerContext
    ) -> ActionResponse:
        """Get the action from the agent."""
        try:
            if self.agent is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("Agent not initialized. Call Init first.")
                return ActionResponse()

            # Get the action from the agent
            # Convert lists back to numpy arrays for the observation
            observation: dict[str, AllowedSerializableTypes] = msgpack.unpackb(
                request.observation, raw=False
            )
            numpy_observation: dict[str, AllowedTypes] = {}
            for key, value in cast(
                gym.spaces.Dict, self.observation_space
            ).spaces.items():
                numpy_observation[key] = native_to_numpy(observation[key], value)
            action = self.agent.get_action(numpy_observation)

            # Convert numpy arrays to lists for serialization
            serializable_action = numpy_to_native(action)

            # Serialize the action
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


def create_agent_server(agent_class: type[CRLAgent], port: int = 50051) -> None:
    """Start the gRPC server."""
    logger = logging.getLogger(__name__)
    agent_server = AgentServicer(agent_class)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_AgentServiceServicer_to_server(agent_server, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Agent server started, listening on port {port}")
    server.wait_for_termination()
