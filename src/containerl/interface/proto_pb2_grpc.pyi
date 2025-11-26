import typing as _typing

import grpc as _grpc

from . import proto_pb2 as proto__pb2

class EnvironmentServiceStub:
    def __init__(self, channel: _grpc.Channel) -> None: ...
    Init: _typing.Callable[[proto__pb2.EnvInitRequest], proto__pb2.EnvInitResponse]
    Reset: _typing.Callable[[proto__pb2.ResetRequest], proto__pb2.ResetResponse]
    Step: _typing.Callable[[proto__pb2.StepRequest], proto__pb2.StepResponse]
    Render: _typing.Callable[[proto__pb2.Empty], proto__pb2.RenderResponse]
    Close: _typing.Callable[[proto__pb2.Empty], proto__pb2.Empty]

class AgentServiceStub:
    def __init__(self, channel: _grpc.Channel) -> None: ...
    Init: _typing.Callable[[proto__pb2.AgentInitRequest], proto__pb2.AgentInitResponse]
    GetAction: _typing.Callable[
        [proto__pb2.ObservationRequest], proto__pb2.ActionResponse
    ]

# New/expanded stubs for servicers and helpers
class EnvironmentServiceServicer:
    def Init(
        self, request: proto__pb2.EnvInitRequest, context: _grpc.ServicerContext
    ) -> proto__pb2.EnvInitResponse: ...
    def Reset(
        self, request: proto__pb2.ResetRequest, context: _grpc.ServicerContext
    ) -> proto__pb2.ResetResponse: ...
    def Step(
        self, request: proto__pb2.StepRequest, context: _grpc.ServicerContext
    ) -> proto__pb2.StepResponse: ...
    def Render(
        self, request: proto__pb2.Empty, context: _grpc.ServicerContext
    ) -> proto__pb2.RenderResponse: ...
    def Close(
        self, request: proto__pb2.Empty, context: _grpc.ServicerContext
    ) -> proto__pb2.Empty: ...

def add_EnvironmentServiceServicer_to_server(
    servicer: EnvironmentServiceServicer, server: _grpc.Server
) -> None: ...

class AgentServiceServicer:
    def Init(
        self, request: proto__pb2.AgentInitRequest, context: _grpc.ServicerContext
    ) -> proto__pb2.AgentInitResponse: ...
    def GetAction(
        self, request: proto__pb2.ObservationRequest, context: _grpc.ServicerContext
    ) -> proto__pb2.ActionResponse: ...

def add_AgentServiceServicer_to_server(
    servicer: AgentServiceServicer, server: _grpc.Server
) -> None: ...
