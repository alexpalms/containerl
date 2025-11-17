import typing as _typing

import grpc as _grpc

from . import proto_pb2 as proto__pb2

class EnvironmentServiceStub:
    def __init__(self, channel: _grpc.Channel) -> None: ...
    Init: _typing.Callable[[proto__pb2.InitRequest], proto__pb2.SpacesResponse]
    Reset: _typing.Callable[[proto__pb2.ResetRequest], proto__pb2.ResetResponse]
    Step: _typing.Callable[[proto__pb2.StepRequest], proto__pb2.StepResponse]
    Render: _typing.Callable[[proto__pb2.Empty], proto__pb2.RenderResponse]
    Close: _typing.Callable[[proto__pb2.Empty], proto__pb2.Empty]

class AgentServiceStub:
    def __init__(self, channel: _grpc.Channel) -> None: ...
    GetSpaces: _typing.Callable[[proto__pb2.Empty], proto__pb2.SpacesResponse]
    GetAction: _typing.Callable[
        [proto__pb2.ObservationRequest], proto__pb2.ActionResponse
    ]

# New/expanded stubs for servicers and helpers
class EnvironmentServiceServicer:
    def Init(self, request: proto__pb2.InitRequest, context: _grpc.ServicerContext) -> proto__pb2.SpacesResponse: ...
    def Reset(self, request: proto__pb2.ResetRequest, context: _grpc.ServicerContext) -> proto__pb2.ResetResponse: ...
    def Step(self, request: proto__pb2.StepRequest, context: _grpc.ServicerContext) -> proto__pb2.StepResponse: ...
    def Render(self, request: proto__pb2.Empty, context: _grpc.ServicerContext) -> proto__pb2.RenderResponse: ...
    def Close(self, request: proto__pb2.Empty, context: _grpc.ServicerContext) -> proto__pb2.Empty: ...

def add_EnvironmentServiceServicer_to_server(servicer: EnvironmentServiceServicer, server: _grpc.Server) -> None: ...

class EnvironmentService:
    @staticmethod
    def Init(
        request: proto__pb2.InitRequest,
        target: str,
        options: _typing.Iterable[_typing.Tuple[str, _typing.Any]] = ...,
        channel_credentials: _typing.Optional[_grpc.ChannelCredentials] = ...,
        call_credentials: _typing.Optional[_grpc.CallCredentials] = ...,
        insecure: bool | None = ...,
        compression: _typing.Optional[_typing.Any] = ...,
        wait_for_ready: _typing.Optional[bool] = ...,
        timeout: _typing.Optional[float] = ...,
        metadata: _typing.Optional[_typing.Iterable[_typing.Tuple[str, str]]] = ...,
    ) -> _grpc.UnaryUnaryMultiCallable: ...

    @staticmethod
    def Reset(...) -> _grpc.UnaryUnaryMultiCallable: ...
    @staticmethod
    def Step(...) -> _grpc.UnaryUnaryMultiCallable: ...
    @staticmethod
    def Render(...) -> _grpc.UnaryUnaryMultiCallable: ...
    @staticmethod
    def Close(...) -> _grpc.UnaryUnaryMultiCallable: ...

class AgentServiceServicer:
    def GetSpaces(self, request: proto__pb2.Empty, context: _grpc.ServicerContext) -> proto__pb2.SpacesResponse: ...
    def GetAction(self, request: proto__pb2.ObservationRequest, context: _grpc.ServicerContext) -> proto__pb2.ActionResponse: ...

def add_AgentServiceServicer_to_server(servicer: AgentServiceServicer, server: _grpc.Server) -> None: ...

class AgentService:
    @staticmethod
    def GetSpaces(
        request: proto__pb2.Empty,
        target: str,
        options: _typing.Iterable[_typing.Tuple[str, _typing.Any]] = ...,
        channel_credentials: _typing.Optional[_grpc.ChannelCredentials] = ...,
        call_credentials: _typing.Optional[_grpc.CallCredentials] = ...,
        insecure: bool | None = ...,
        compression: _typing.Optional[_typing.Any] = ...,
        wait_for_ready: _typing.Optional[bool] = ...,
        timeout: _typing.Optional[float] = ...,
        metadata: _typing.Optional[_typing.Iterable[_typing.Tuple[str, str]]] = ...,
    ) -> _grpc.UnaryUnaryMultiCallable: ...

    @staticmethod
    def GetAction(...) -> _grpc.UnaryUnaryMultiCallable: ...
