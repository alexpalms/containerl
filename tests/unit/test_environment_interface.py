"""Unit tests for environment interface client and server factory with type hints."""
# ruff: noqa: N802

from types import SimpleNamespace
from typing import Any, cast

import grpc
import gymnasium.spaces as spaces
import msgpack
import numpy as np
import pytest

from containerl.interface.environment import client as client_mod
from containerl.interface.environment import client_vec as client_vec_mod
from containerl.interface.environment import server_factory as sf_mod
from containerl.interface.environment import server_factory_vec as sf_vec_mod

pytestmark = pytest.mark.unit


class DummyContext:
    def __init__(self) -> None:
        self.code: Any | None = None
        self.details: str | None = None

    def set_code(self, code: Any) -> None:
        self.code = code

    def set_details(self, details: str) -> None:
        self.details = details


def test_environment_servicer_init_success() -> None:
    class MyEnv:  # simple stand-in for Env
        def __init__(self, **kwargs: Any) -> None:
            self.observation_space: spaces.Space[Any] = spaces.Dict(
                {"obs": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)}
            )
            self.action_space: spaces.Space[Any] = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            )
            self.init_info = {"k": "v"}
            self.render_mode = None

        def reset(
            self, seed: Any = None, options: Any = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            return ({"obs": np.array([0.0], dtype=np.float32)}, {})

        def step(
            self, action: Any
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            return ({"obs": np.array([0.0], dtype=np.float32)}, 0.0, False, False, {})

        def render(self) -> Any:
            return None

        def close(self) -> None:  # pragma: no cover
            return None

    servicer = sf_mod.EnvironmentServer(cast(Any, MyEnv))

    class Req:
        def __init__(self, init_args: dict[str, Any] | None = None) -> None:
            self._init_args = init_args

        def HasField(self, name: str) -> bool:  # noqa: N802
            return name == "init_args" and self._init_args is not None

        @property
        def init_args(self) -> bytes:
            return msgpack.packb(self._init_args, use_bin_type=True)

    ctx = DummyContext()
    resp = servicer.Init(cast(Any, Req({"a": 1})), cast(Any, ctx))
    info = msgpack.unpackb(resp.info, raw=False)
    assert info == {"k": "v"}
    assert "obs" in resp.observation_space


def test_environment_servicer_init_bad_obs_space_sets_error() -> None:
    class BadEnv:  # simple stand-in for Env
        def __init__(self, **kwargs: Any) -> None:
            self.observation_space = spaces.Box(low=0, high=1, shape=(1,))
            self.action_space = spaces.Discrete(2)

        def reset(
            self, seed: Any = None, options: Any = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            return ({}, {})

        def step(
            self, action: Any
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            return ({}, 0.0, False, False, {})

    servicer = sf_mod.EnvironmentServer(cast(Any, BadEnv))

    class Req:
        def HasField(self, name: str) -> bool:
            return False

    ctx = DummyContext()
    _ = servicer.Init(cast(Any, Req()), cast(Any, ctx))
    assert ctx.code == grpc.StatusCode.INTERNAL
    assert ctx.details is not None and "Observation space must be a Dict" in ctx.details


def test_reset_and_step_precondition_and_success() -> None:
    class SimpleEnv:  # simple stand-in for Env
        def __init__(self) -> None:
            self.observation_space = spaces.Dict(
                {"obs": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)}
            )
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            self.render_mode = None

        def reset(
            self, seed: Any = None, options: Any = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            return ({"obs": np.array([0.0], dtype=np.float32)}, {})

        def step(
            self, action: Any
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            return ({"obs": np.array([0.0], dtype=np.float32)}, 0.1, True, False, {})

        def render(self) -> Any:
            return None

    servicer = sf_mod.EnvironmentServer(cast(Any, SimpleEnv))
    ctx = DummyContext()

    # Reset before init should fail
    # Reset/Step requests should implement HasField to mimic proto messages
    class ProtoReq(SimpleNamespace):
        def HasField(self, name: str) -> bool:  # noqa: N802
            return False

    reset_req = ProtoReq()
    _ = servicer.Reset(cast(Any, reset_req), cast(Any, ctx))
    assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION

    # Step before init should fail
    step_req = SimpleNamespace()
    _ = servicer.Step(cast(Any, step_req), cast(Any, ctx))
    assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION

    # Now initialize and test Reset and Step
    class Req:
        def HasField(self, name: str) -> bool:
            return False

    servicer.Init(cast(Any, Req()), cast(Any, ctx))

    # use Req-like object for reset calls that implement HasField where server expects it
    class ResetReq:
        def HasField(self, name: str) -> bool:  # noqa: N802
            return False

    reset_resp = servicer.Reset(cast(Any, ResetReq()), cast(Any, ctx))
    # Debug: ensure observation bytes are valid msgpack
    try:
        obs = msgpack.unpackb(reset_resp.observation, raw=False)
    except Exception as e:
        raise RuntimeError(
            f"Unpack failed: {e}; bytes={reset_resp.observation!r}; ctx.code={ctx.code}; ctx.details={ctx.details!r}"
        )
    assert "obs" in obs

    # Step
    fake_action = msgpack.packb([0.0], use_bin_type=True)
    step_req2 = SimpleNamespace(action=fake_action)
    step_resp = servicer.Step(cast(Any, step_req2), cast(Any, ctx))
    sobs = msgpack.unpackb(step_resp.observation, raw=False)
    assert "obs" in sobs


def test_render_and_close() -> None:
    class ImgEnv:  # simple stand-in for Env
        def __init__(self) -> None:
            self.observation_space = spaces.Dict(
                {"obs": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)}
            )
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            self.render_mode = "rgb_array"

        def reset(
            self, seed: Any = None, options: Any = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            return ({"obs": np.array([0.0], dtype=np.float32)}, {})

        def step(
            self, action: Any
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            return ({"obs": np.array([0.0], dtype=np.float32)}, 0.0, False, False, {})

        def render(self) -> Any:
            # return a 3D numpy array (H,W,3)
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self) -> None:  # pragma: no cover
            return None

    servicer = sf_mod.EnvironmentServer(cast(Any, ImgEnv))
    ctx = DummyContext()

    # Use request object with HasField to avoid internal errors
    class InitReq:
        def HasField(self, name: str) -> bool:  # noqa: N802
            return False

    servicer.Init(cast(Any, InitReq()), cast(Any, ctx))
    render_resp = servicer.Render(cast(Any, SimpleNamespace()), cast(Any, ctx))
    assert render_resp.render_data

    # Close should return Empty
    close_resp = servicer.Close(cast(Any, SimpleNamespace()), cast(Any, ctx))
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

    def fake_channel_ready_future(ch: Any) -> FakeFuture:
        return FakeFuture(ok=True)

    monkeypatch.setattr(grpc, "channel_ready_future", fake_channel_ready_future)

    # Fake stub with Init, Reset, Step, Render, Close
    class FakeStub:
        def __init__(self, channel: Any) -> None:
            pass

        def Init(self, req: Any) -> Any:
            proto_box = SimpleNamespace(
                type="Box", low=[0.0], high=[1.0], shape=[1], dtype="float32"
            )
            action_proto = SimpleNamespace(
                type="Box", low=[-1.0], high=[1.0], shape=[1], dtype="float32"
            )
            return SimpleNamespace(
                observation_space={"obs": proto_box},
                action_space=action_proto,
                info=msgpack.packb({"x": 1}, use_bin_type=True),
                num_envs=1,
                environment_type=0,
                render_mode="None",
            )

        def Reset(self, req: Any) -> Any:
            return SimpleNamespace(
                observation=msgpack.packb({"obs": [0.0]}, use_bin_type=True),
                info=msgpack.packb({}, use_bin_type=True),
            )

        def Step(self, req: Any) -> Any:
            return SimpleNamespace(
                observation=msgpack.packb({"obs": [0.0]}, use_bin_type=True),
                reward=msgpack.packb(0.5, use_bin_type=True),
                terminated=msgpack.packb(True, use_bin_type=True),
                truncated=msgpack.packb(False, use_bin_type=True),
                info=msgpack.packb({}, use_bin_type=True),
            )

        def Render(self, req: Any) -> Any:
            array = np.zeros((2, 2, 3), dtype=np.uint8)
            data = {
                "shape": array.shape,
                "dtype": str(array.dtype),
                "data": array.tobytes(),
            }
            return SimpleNamespace(render_data=msgpack.packb(data, use_bin_type=True))

        def Close(self, req: Any) -> Any:
            return SimpleNamespace()

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
    class VecEnv:  # simple stand-in for CRLVecGymEnvironment
        def __init__(self) -> None:
            self.num_envs = 2
            self.observation_space = spaces.Dict(
                {"obs": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)}
            )
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.init_info = [{"k": "v1"}, {"k": "v2"}]
            self.render_mode = None

        def reset(
            self, seed: Any = None, options: Any = None
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            return (
                {
                    "obs": np.stack(
                        [
                            np.array([0.0], dtype=np.float32)
                            for _ in range(self.num_envs)
                        ]
                    )
                },
                [{}, {}],
            )

        def step(
            self, action: Any
        ) -> tuple[
            dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]
        ]:
            obs = {
                "obs": np.stack(
                    [np.array([0.0], dtype=np.float32) for _ in range(self.num_envs)]
                )
            }
            reward = np.array([0.1, 0.2], dtype=np.float32)
            terminated = np.array([False, False], dtype=bool)
            truncated = np.array([False, False], dtype=bool)
            info: list[dict[str, Any]] = [{}, {}]
            return (obs, reward, terminated, truncated, info)

        def render(self) -> Any:
            return None

        def close(self) -> None:  # pragma: no cover
            return None

    servicer = sf_vec_mod.VecEnvironmentServicer(cast(Any, VecEnv))

    class Req:
        def HasField(self, name: str) -> bool:
            return False

    ctx = DummyContext()
    resp = servicer.Init(cast(Any, Req()), cast(Any, ctx))
    assert resp.num_envs == 2

    # Reset/Step before init should fail on a fresh servicer
    servicer2 = sf_vec_mod.VecEnvironmentServicer(cast(Any, VecEnv))
    _ = servicer2.Reset(cast(Any, SimpleNamespace()), cast(Any, ctx))
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

    def _channel_ready_future(ch: Any) -> FakeFuture:  # pragma: no cover
        return FakeFuture()

    monkeypatch.setattr(grpc, "channel_ready_future", _channel_ready_future)

    class FakeStub:
        def __init__(self, channel: Any) -> None:
            pass

        def Init(self, req: Any) -> Any:
            proto_box = SimpleNamespace(
                type="Box", low=[0.0, 0.0], high=[1.0, 1.0], shape=[2], dtype="float32"
            )
            action_proto = SimpleNamespace(
                type="Box",
                low=[-1.0, -1.0],
                high=[1.0, 1.0],
                shape=[2],
                dtype="float32",
            )
            return SimpleNamespace(
                observation_space={"obs": proto_box},
                action_space=action_proto,
                info=msgpack.packb([{"x": 1}, {"x": 2}], use_bin_type=True),
                num_envs=2,
                environment_type=1,
                render_mode="None",
            )

        def Reset(self, req: Any) -> Any:
            return SimpleNamespace(
                observation=msgpack.packb(
                    {"obs": [[0.0, 0.0], [0.0, 0.0]]}, use_bin_type=True
                ),
                info=msgpack.packb([{}, {}], use_bin_type=True),
            )

        def Step(self, req: Any) -> Any:
            return SimpleNamespace(
                observation=msgpack.packb(
                    {"obs": [[0.0, 0.0], [0.0, 0.0]]}, use_bin_type=True
                ),
                reward=msgpack.packb([0.1, 0.2], use_bin_type=True),
                terminated=msgpack.packb([False, False], use_bin_type=True),
                truncated=msgpack.packb([False, False], use_bin_type=True),
                info=msgpack.packb([{}, {}], use_bin_type=True),
            )

        def Render(self, req: Any) -> Any:
            array = np.zeros((2, 2, 3), dtype=np.uint8)
            data = {
                "shape": array.shape,
                "dtype": str(array.dtype),
                "data": array.tobytes(),
            }
            return SimpleNamespace(render_data=msgpack.packb(data, use_bin_type=True))

        def Close(self, req: Any) -> Any:
            return SimpleNamespace()

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
