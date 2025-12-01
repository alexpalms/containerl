"""Unit tests for agent interface client and server factory with type hints."""

from typing import cast

import grpc
import gymnasium.spaces as spaces
import msgpack
import numpy as np
import pytest
from google.protobuf import message as _message

from containerl import (
    ActionResponse,
    AgentInitRequest,
    AgentInitResponse,
    AllowedInfoValueTypes,
    AllowedTypes,
    ObservationRequest,
    Space,
)
from containerl.interface.agent import client as client_mod
from containerl.interface.agent import server_factory as sf_mod

pytestmark = pytest.mark.unit


class DummyContext:
    def __init__(self) -> None:
        self.code: grpc.StatusCode | None = None
        self.details: str | None = None

    def set_code(self, code: grpc.StatusCode) -> None:
        self.code = code

    def set_details(self, details: str) -> None:
        self.details = details


def test_agent_servicer_init_success() -> None:
    class MyAgent(sf_mod.CRLAgent):
        def __init__(self, **kwargs: dict[str, AllowedInfoValueTypes]) -> None:
            self.observation_space: spaces.Space[dict[str, AllowedTypes]] = spaces.Dict(
                {"obs": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)}
            )
            self.action_space: spaces.Space[AllowedTypes] = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            )
            self.init_info: dict[str, AllowedInfoValueTypes] = {"k": "v"}

        def get_action(self, observation: dict[str, AllowedTypes]) -> AllowedTypes:
            return np.array([0.1]).astype(np.float32)

    servicer = sf_mod.AgentServicer(MyAgent)

    # request object with HasField and packed init_args
    class Req(_message.Message):
        def __init__(
            self, init_args: dict[str, AllowedInfoValueTypes] | None = None
        ) -> None:
            self._init_args = init_args

        def HasField(self, field_name: str) -> bool:  # noqa: N802
            return field_name == "init_args" and self._init_args is not None

        @property
        def init_args(self) -> bytes:
            return msgpack.packb(self._init_args, use_bin_type=True)

    ctx = cast(grpc.ServicerContext, DummyContext())
    req = cast(AgentInitRequest, Req({"a": 1}))
    resp = servicer.Init(req, ctx)
    # info should unpack to the init_info dict
    info = msgpack.unpackb(resp.info, raw=False)
    assert info == {"k": "v"}
    # observation_space should contain 'obs'
    assert "obs" in resp.observation_space


def test_agent_servicer_init_bad_obs_space_sets_error() -> None:
    class BadAgent(sf_mod.CRLAgent):
        def __init__(self, **kwargs: dict[str, AllowedInfoValueTypes]) -> None:
            self.observation_space = spaces.Box(  # type: ignore
                low=0, high=1, shape=(1,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(2)

        def get_action(self, observation: dict[str, AllowedTypes]) -> AllowedTypes:
            return np.int8(0)

    servicer = sf_mod.AgentServicer(BadAgent)

    class Req(_message.Message):
        def HasField(self, field_name: str) -> bool:  # noqa: N802
            return False

    ctx = DummyContext()
    req = cast(AgentInitRequest, Req())
    _ = servicer.Init(req, cast(grpc.ServicerContext, ctx))
    assert ctx.code == grpc.StatusCode.INTERNAL
    assert ctx.details is not None and "Observation space must be a Dict" in ctx.details


def test_get_action_not_initialized_returns_failed_precondition() -> None:
    class DummyConcrete(sf_mod.CRLAgent):
        def get_action(self, observation: dict[str, AllowedTypes]) -> AllowedTypes:
            raise NotImplementedError

    servicer = sf_mod.AgentServicer(DummyConcrete)  # agent not created yet

    class Req(_message.Message):
        def __init__(self) -> None:
            self.observation: bytes = msgpack.packb({"obs": [0.0]}, use_bin_type=True)

    ctx = DummyContext()
    req = cast(ObservationRequest, Req())
    _ = servicer.GetAction(req, cast(grpc.ServicerContext, ctx))
    assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION
    assert ctx.details is not None and "Agent not initialized" in ctx.details


def test_get_action_success_and_error() -> None:
    class MyAgent(sf_mod.CRLAgent):
        def __init__(self) -> None:
            self.observation_space: spaces.Space[dict[str, AllowedTypes]] = spaces.Dict(
                {"obs": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)}
            )
            self.action_space: spaces.Space[AllowedTypes] = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            )

        def get_action(self, observation: dict[str, AllowedTypes]) -> AllowedTypes:
            return np.array([0.9])

    servicer = sf_mod.AgentServicer(MyAgent)
    # manually set agent and observation space as Init would
    servicer.agent = MyAgent()
    servicer.observation_space = servicer.agent.observation_space

    class Req(_message.Message):
        def __init__(self) -> None:
            self.observation: bytes = msgpack.packb({"obs": [0.0]}, use_bin_type=True)

    ctx = cast(grpc.ServicerContext, DummyContext())
    req = cast(ObservationRequest, Req())
    resp = servicer.GetAction(req, ctx)
    # unpack action from response
    action = msgpack.unpackb(resp.action, raw=False)
    assert isinstance(action, list) or isinstance(action, (int, float))

    # Now simulate agent raising an exception
    def bad_action(_: dict[str, AllowedTypes]) -> None:
        raise RuntimeError("boom")

    servicer.agent.get_action = bad_action  # type: ignore[assignment]
    ctx2 = DummyContext()
    req2 = cast(ObservationRequest, Req())
    servicer.GetAction(req2, cast(grpc.ServicerContext, ctx2))
    assert ctx2.code == grpc.StatusCode.INTERNAL
    assert ctx2.details is not None and "Error getting agent action" in ctx2.details


def test_agent_client_init_and_get_action(monkeypatch: pytest.MonkeyPatch) -> None:
    # Monkeypatch grpc channel and channel_ready_future
    class FakeChannel:
        def close(self) -> None:
            self.closed = True

    def fake_insecure_channel(addr: str) -> FakeChannel:
        return FakeChannel()

    class FakeFuture:
        def __init__(self, ok: bool = True) -> None:
            self._ok = ok

        def result(self, timeout: float | None = None) -> None:
            if not self._ok:
                raise grpc.FutureTimeoutError()
            return None

    monkeypatch.setattr(grpc, "insecure_channel", fake_insecure_channel)

    def fake_channel_ready_future(ch: grpc.Channel) -> FakeFuture:
        return FakeFuture(ok=True)

    monkeypatch.setattr(grpc, "channel_ready_future", fake_channel_ready_future)

    # Fake stub with Init and GetAction
    class FakeStub:
        def __init__(self, channel: grpc.Channel) -> None:
            pass

        def Init(self, req: AgentInitRequest) -> AgentInitResponse:  # noqa: N802
            # Return a proto-like object without importing generated proto classes
            proto_box = Space(
                type="Box", low=[0.0], high=[1.0], shape=[1], dtype="float32"
            )
            action_proto = Space(
                type="Box", low=[-1.0], high=[1.0], shape=[1], dtype="float32"
            )
            return AgentInitResponse(
                observation_space={"obs": proto_box},
                action_space=action_proto,
                info=msgpack.packb({"x": 1}, use_bin_type=True),
            )

        def GetAction(self, req: ObservationRequest) -> ActionResponse:  # noqa: N802
            resp = ActionResponse()
            resp.action = msgpack.packb([0.5], use_bin_type=True)
            return resp

    monkeypatch.setattr(client_mod, "AgentServiceStub", FakeStub)

    # Initialize client
    cli: client_mod.AgentClient = client_mod.AgentClient("localhost:1234", timeout=1.0)
    # sample observation compatible with observation_space
    obs = cli.observation_space.sample()
    action = cli.get_action(obs)
    assert isinstance(action, np.ndarray)


def test_agent_client_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeChannel:
        def close(self) -> None:
            pass

    def _fake_insecure_channel(addr: str) -> FakeChannel:
        return FakeChannel()

    class _F:
        @staticmethod
        def result(timeout: float | None = None) -> None:
            raise grpc.FutureTimeoutError()

    def _fake_channel_ready_future(ch: grpc.Channel) -> _F:
        return _F()

    monkeypatch.setattr(grpc, "insecure_channel", _fake_insecure_channel)
    monkeypatch.setattr(grpc, "channel_ready_future", _fake_channel_ready_future)

    # AgentServiceStub won't be reached; ensure TimeoutError is raised
    with pytest.raises(TimeoutError):
        client_mod.AgentClient("localhost:1234", timeout=0.01)


def test_agent_check_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgent(sf_mod.CRLAgent):
        def __init__(
            self,
            server_address: str,
            timeout: float,
            **kwargs: dict[str, AllowedInfoValueTypes],
        ) -> None:
            # accept server_address, timeout and init_args
            self.observation_space: spaces.Space[dict[str, AllowedTypes]] = spaces.Dict(
                {"obs": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)}
            )
            self.action_space: spaces.Space[AllowedTypes] = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            )

        def get_action(self, observation: dict[str, AllowedTypes]) -> np.ndarray:
            return np.array([0.0])

    monkeypatch.setattr(client_mod, "AgentClient", FakeAgent)
    # should run without raising
    client_mod.agent_check("localhost:1234", num_steps=2)
