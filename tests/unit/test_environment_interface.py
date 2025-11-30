"""Unit tests for environment interface client and server factory with type hints."""
# ruff: noqa: N802

from types import SimpleNamespace
from typing import cast

import grpc
import gymnasium.spaces as spaces
import msgpack
import numpy as np
import pytest
from google.protobuf import message as _message
from gymnasium import Env
from numpy.typing import NDArray

from containerl import (
    AllowedInfoValueTypes,
    AllowedTypes,
    Empty,
    EnvInitRequest,
    EnvInitResponse,
    EnvironmentType,
    RenderResponse,
    ResetRequest,
    ResetResponse,
    Space,
    StepRequest,
    StepResponse,
)
from containerl.interface.environment import client as client_mod
from containerl.interface.environment import client_vec as client_vec_mod
from containerl.interface.environment import server_factory as sf_mod
from containerl.interface.environment import server_factory_vec as sf_vec_mod

pytestmark = pytest.mark.unit


class DummyContext:
    def __init__(self) -> None:
        self.code: grpc.StatusCode | None = None
        self.details: str | None = None

    def set_code(self, code: grpc.StatusCode) -> None:
        self.code = code

    def set_details(self, details: str) -> None:
        self.details = details


def test_environment_servicer_init_success() -> None:
    class MyEnv(Env[dict[str, AllowedTypes], AllowedTypes]):  # simple stand-in for Env
        def __init__(self, **kwargs: dict[str, AllowedInfoValueTypes]) -> None:
            self.observation_space: spaces.Space[dict[str, AllowedTypes]] = spaces.Dict(
                {"obs": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)}
            )
            self.action_space: spaces.Space[AllowedTypes] = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            )
            self.init_info = {"k": "v"}
            self.render_mode = None

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, AllowedInfoValueTypes] | None = None,
        ) -> tuple[dict[str, AllowedTypes], dict[str, AllowedInfoValueTypes]]:
            return ({"obs": np.array([0.0], dtype=np.float32)}, {})

        def step(
            self, action: AllowedTypes
        ) -> tuple[
            dict[str, AllowedTypes], float, bool, bool, dict[str, AllowedInfoValueTypes]
        ]:
            return ({"obs": np.array([0.0], dtype=np.float32)}, 0.0, False, False, {})

        def render(self) -> NDArray[np.uint8] | None:  # type: ignore
            return None

        def close(self) -> None:  # pragma: no cover
            return None

    servicer = sf_mod.EnvironmentServer(MyEnv)

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

    ctx = DummyContext()
    resp = servicer.Init(
        cast(EnvInitRequest, Req({"a": 1})), cast(grpc.ServicerContext, ctx)
    )
    info = msgpack.unpackb(resp.info, raw=False)
    assert info == {"k": "v"}
    assert "obs" in resp.observation_space


def test_environment_servicer_init_bad_obs_space_sets_error() -> None:
    class BadEnv(Env[dict[str, AllowedTypes], AllowedTypes]):  # simple stand-in for Env
        def __init__(self, **kwargs: dict[str, AllowedInfoValueTypes]) -> None:
            self.observation_space = spaces.Box(low=0, high=1, shape=(1,))  # type: ignore
            self.action_space = spaces.Discrete(2)

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, AllowedInfoValueTypes] | None = None,
        ) -> tuple[dict[str, AllowedTypes], dict[str, AllowedInfoValueTypes]]:
            return ({}, {})

        def step(
            self, action: AllowedTypes
        ) -> tuple[
            dict[str, AllowedTypes], float, bool, bool, dict[str, AllowedInfoValueTypes]
        ]:
            return ({}, 0.0, False, False, {})

    servicer = sf_mod.EnvironmentServer(BadEnv)

    class Req:
        def HasField(self, name: str) -> bool:
            return False

    ctx = DummyContext()
    _ = servicer.Init(cast(EnvInitRequest, Req()), cast(grpc.ServicerContext, ctx))
    assert ctx.code == grpc.StatusCode.INTERNAL
    assert ctx.details is not None and "Observation space must be a Dict" in ctx.details


def test_reset_and_step_precondition_and_success() -> None:
    class SimpleEnv(
        Env[dict[str, AllowedTypes], AllowedTypes]
    ):  # simple stand-in for Env
        def __init__(self) -> None:
            self.observation_space = spaces.Dict(
                {"obs": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)}
            )
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            self.render_mode = None

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, AllowedInfoValueTypes] | None = None,
        ) -> tuple[dict[str, AllowedTypes], dict[str, AllowedInfoValueTypes]]:
            return ({"obs": np.array([0.0], dtype=np.float32)}, {})

        def step(
            self, action: AllowedTypes
        ) -> tuple[
            dict[str, AllowedTypes], float, bool, bool, dict[str, AllowedInfoValueTypes]
        ]:
            return ({"obs": np.array([0.0], dtype=np.float32)}, 0.1, True, False, {})

        def render(self) -> NDArray[np.uint8] | None:  # type: ignore
            return None

    servicer = sf_mod.EnvironmentServer(SimpleEnv)
    ctx = DummyContext()

    # Reset before init should fail
    # Reset/Step requests should implement HasField to mimic proto messages
    class ProtoReq(SimpleNamespace):
        def HasField(self, name: str) -> bool:  # noqa: N802
            return False

    reset_req = ProtoReq()
    _ = servicer.Reset(cast(ResetRequest, reset_req), cast(grpc.ServicerContext, ctx))
    assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION

    # Step before init should fail
    step_req = SimpleNamespace()
    _ = servicer.Step(cast(StepRequest, step_req), cast(grpc.ServicerContext, ctx))
    assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION

    # Now initialize and test Reset and Step
    class Req:
        def HasField(self, name: str) -> bool:
            return False

    servicer.Init(cast(EnvInitRequest, Req()), cast(grpc.ServicerContext, ctx))

    # use Req-like object for reset calls that implement HasField where server expects it
    class ResetReq:
        def HasField(self, name: str) -> bool:  # noqa: N802
            return False

    reset_resp = servicer.Reset(
        cast(ResetRequest, ResetReq()), cast(grpc.ServicerContext, ctx)
    )
    # Debug: ensure observation bytes are valid msgpack
    try:
        obs = msgpack.unpackb(reset_resp.observation, raw=False)
    except Exception as e:
        raise RuntimeError(
            f"Unpack failed: {e}; bytes={reset_resp.observation!r}; ctx.code={ctx.code}; ctx.details={ctx.details!r}"
        ) from e
    assert "obs" in obs

    # Step
    fake_action = msgpack.packb([0.0], use_bin_type=True)
    step_req2 = SimpleNamespace(action=fake_action)
    step_resp = servicer.Step(
        cast(StepRequest, step_req2), cast(grpc.ServicerContext, ctx)
    )
    sobs = msgpack.unpackb(step_resp.observation, raw=False)
    assert "obs" in sobs


def test_render_and_close() -> None:
    class ImgEnv(Env[dict[str, AllowedTypes], AllowedTypes]):  # simple stand-in for Env
        def __init__(self) -> None:
            self.observation_space = spaces.Dict(
                {"obs": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)}
            )
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            self.render_mode = "rgb_array"

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, AllowedInfoValueTypes] | None = None,
        ) -> tuple[dict[str, AllowedTypes], dict[str, AllowedInfoValueTypes]]:
            return ({"obs": np.array([0.0], dtype=np.float32)}, {})

        def step(
            self, action: AllowedTypes
        ) -> tuple[
            dict[str, AllowedTypes], float, bool, bool, dict[str, AllowedInfoValueTypes]
        ]:
            return ({"obs": np.array([0.0], dtype=np.float32)}, 0.0, False, False, {})

        def render(self) -> NDArray[np.uint8]:  # type: ignore
            # return a 3D numpy array (H,W,3)
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self) -> None:  # pragma: no cover
            return None

    servicer = sf_mod.EnvironmentServer(ImgEnv)
    ctx = DummyContext()

    # Use request object with HasField to avoid internal errors
    class InitReq(_message.Message):
        def HasField(self, field_name: str) -> bool:  # noqa: N802
            return False

    servicer.Init(cast(EnvInitRequest, InitReq()), cast(grpc.ServicerContext, ctx))
    render_resp = servicer.Render(Empty(), cast(grpc.ServicerContext, ctx))
    assert render_resp.render_data

    # Close should return Empty
    close_resp = servicer.Close(Empty(), cast(grpc.ServicerContext, ctx))
    assert close_resp is not None


def test_client_and_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake grpc channel and readiness
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

    # Fake stub with Init, Reset, Step, Render, Close
    class FakeStub:
        def __init__(self, channel: grpc.Channel) -> None:
            pass

        def Init(self, req: EnvInitRequest) -> EnvInitResponse:
            proto_box = Space(
                type="Box", low=[0.0], high=[1.0], shape=[1], dtype="float32"
            )
            action_proto = Space(
                type="Box", low=[-1.0], high=[1.0], shape=[1], dtype="float32"
            )
            return EnvInitResponse(
                observation_space={"obs": proto_box},
                action_space=action_proto,
                info=msgpack.packb({"x": 1}, use_bin_type=True),
                num_envs=1,
                environment_type=EnvironmentType.STANDARD,
                render_mode="None",
            )

        def Reset(self, req: ResetRequest) -> ResetResponse:
            return ResetResponse(
                observation=msgpack.packb({"obs": [0.0]}, use_bin_type=True),
                info=msgpack.packb({}, use_bin_type=True),
            )

        def Step(self, req: StepRequest) -> StepResponse:
            return StepResponse(
                observation=msgpack.packb({"obs": [0.0]}, use_bin_type=True),
                reward=msgpack.packb(0.5, use_bin_type=True),
                terminated=msgpack.packb(True, use_bin_type=True),
                truncated=msgpack.packb(False, use_bin_type=True),
                info=msgpack.packb({}, use_bin_type=True),
            )

        def Render(self, req: Empty) -> RenderResponse:
            array = np.zeros((2, 2, 3), dtype=np.uint8)
            data = {
                "shape": array.shape,
                "dtype": str(array.dtype),
                "data": array.tobytes(),
            }
            return RenderResponse(render_data=msgpack.packb(data, use_bin_type=True))

        def Close(self, req: Empty) -> Empty:
            return Empty()

    monkeypatch.setattr(client_mod, "EnvironmentServiceStub", FakeStub)

    # Initialize client
    cli: client_mod.CRLEnvironmentClient = client_mod.CRLEnvironmentClient(
        "localhost:1234", timeout=1.0
    )
    obs, info = cli.reset()
    assert "obs" in obs
    _ = info
    out = cli.step(cli.action_space.sample())
    assert isinstance(out[1], float) or hasattr(out[1], "__float__")
    frame = cli.render()
    assert frame is not None and isinstance(frame, np.ndarray)
    cli.close()

    # Adapter
    adapter = client_mod.CRLGymEnvironmentAdapter("localhost:1234", timeout=1.0)
    aobs, ainfo = adapter.reset()
    assert "obs" in aobs
    _ = ainfo
    adapter.step(adapter.action_space.sample())
    adapter.render()
    adapter.close()


def test_vectorized_servicer_and_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class VecEnv(
        sf_vec_mod.CRLVecGymEnvironment
    ):  # simple stand-in for CRLVecGymEnvironment
        def __init__(self) -> None:
            self.num_envs = 2
            self.observation_space = spaces.Dict(
                {"obs": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)}
            )
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.init_info = {"k1": "v1", "k2": "v2"}
            self.render_mode = None

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, AllowedInfoValueTypes] | None = None,
        ) -> tuple[
            dict[str, NDArray[np.floating | np.integer]],
            list[dict[str, AllowedInfoValueTypes]],
        ]:
            return (
                {
                    "obs": np.stack(
                        [
                            np.array([0.0], dtype=np.floating)
                            for _ in range(self.num_envs)
                        ]
                    )
                },
                [{}, {}],
            )

        def step(
            self, action: NDArray[np.floating | np.integer]
        ) -> tuple[
            dict[str, NDArray[np.floating | np.integer]],
            NDArray[np.floating],
            NDArray[np.bool_],
            NDArray[np.bool_],
            list[dict[str, AllowedInfoValueTypes]],
        ]:
            obs = cast(
                dict[str, NDArray[np.floating | np.integer]],
                {
                    "obs": np.stack(
                        [
                            np.array([0.0], dtype=np.floating)
                            for _ in range(self.num_envs)
                        ]
                    )
                },
            )
            reward = np.array([0.1, 0.2], dtype=np.floating)
            terminated = np.array([False, False], dtype=np.bool_)
            truncated = np.array([False, False], dtype=np.bool_)
            info: list[dict[str, AllowedInfoValueTypes]] = [{}, {}]
            return (obs, reward, terminated, truncated, info)

        def render(self) -> NDArray[np.uint8] | None:  # type: ignore
            return None

        def close(self) -> None:  # pragma: no cover
            return None

    servicer = sf_vec_mod.VecEnvironmentServicer(VecEnv)

    class Req:
        def HasField(self, name: str) -> bool:
            return False

    ctx = DummyContext()
    resp = servicer.Init(cast(EnvInitRequest, Req()), cast(grpc.ServicerContext, ctx))
    assert resp.num_envs == 2

    # Reset/Step before init should fail on a fresh servicer
    servicer2 = sf_vec_mod.VecEnvironmentServicer(VecEnv)
    _ = servicer2.Reset(ResetRequest(), cast(grpc.ServicerContext, ctx))
    assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION

    # Client and vec adapter
    class FakeChannel:
        def close(self) -> None:  # pragma: no cover
            self.closed = True

    def fake_insecure_channel(addr: str) -> FakeChannel:
        return FakeChannel()

    class FakeFuture:
        def result(self, timeout: float | None = None) -> None:  # pragma: no cover
            return None

    monkeypatch.setattr(grpc, "insecure_channel", fake_insecure_channel)

    def _channel_ready_future(ch: grpc.Channel) -> FakeFuture:  # pragma: no cover
        return FakeFuture()

    monkeypatch.setattr(grpc, "channel_ready_future", _channel_ready_future)

    class FakeStub:
        def __init__(self, channel: grpc.Channel) -> None:
            pass

        def Init(self, req: EnvInitRequest) -> EnvInitResponse:
            proto_box = Space(
                type="Box", low=[0.0, 0.0], high=[1.0, 1.0], shape=[2], dtype="float32"
            )
            action_proto = Space(
                type="Box",
                low=[-1.0, -1.0],
                high=[1.0, 1.0],
                shape=[2],
                dtype="float32",
            )
            return EnvInitResponse(
                observation_space={"obs": proto_box},
                action_space=action_proto,
                info=msgpack.packb([{"x": 1}, {"x": 2}], use_bin_type=True),
                num_envs=2,
                environment_type=EnvironmentType.VECTORIZED,
                render_mode="None",
            )

        def Reset(self, req: ResetRequest) -> ResetResponse:
            return ResetResponse(
                observation=msgpack.packb(
                    {"obs": [[0.0, 0.0], [0.0, 0.0]]}, use_bin_type=True
                ),
                info=msgpack.packb([{}, {}], use_bin_type=True),
            )

        def Step(self, req: StepRequest) -> StepResponse:
            return StepResponse(
                observation=msgpack.packb(
                    {"obs": [[0.0, 0.0], [0.0, 0.0]]}, use_bin_type=True
                ),
                reward=msgpack.packb([0.1, 0.2], use_bin_type=True),
                terminated=msgpack.packb([False, False], use_bin_type=True),
                truncated=msgpack.packb([False, False], use_bin_type=True),
                info=msgpack.packb([{}, {}], use_bin_type=True),
            )

        def Render(self, req: Empty) -> RenderResponse:
            array = np.zeros((2, 2, 3), dtype=np.uint8)
            data = {
                "shape": array.shape,
                "dtype": str(array.dtype),
                "data": array.tobytes(),
            }
            return RenderResponse(render_data=msgpack.packb(data, use_bin_type=True))

        def Close(self, req: Empty) -> Empty:
            return Empty()

    monkeypatch.setattr(client_vec_mod, "EnvironmentServiceStub", FakeStub)
    # silence unused ainfo variable
    pass

    cli_vec: client_vec_mod.CRLVecEnvironmentClient = (
        client_vec_mod.CRLVecEnvironmentClient("localhost:1234", timeout=1.0)
    )
    nav, ninfo = cli_vec.reset()
    assert "obs" in nav
    _ = ninfo
    out = cli_vec.step(
        np.stack([cli_vec.action_space.sample() for _ in range(cli_vec.num_envs)])
    )
    assert out[1].shape[0] == 2
    frame = cli_vec.render()
    assert frame is not None and isinstance(frame, np.ndarray)
    cli_vec.close()

    # Adapter
    adapter = client_vec_mod.CRLVecGymEnvironmentAdapter("localhost:1234", timeout=1.0)
    aobs, ainfo = adapter.reset()
    assert "obs" in aobs
    _ = ainfo
    adapter.step(
        np.stack([adapter.action_space.sample() for _ in range(cli_vec.num_envs)])
    )
    adapter.render()
    adapter.close()
