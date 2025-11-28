"""Unit tests for containerl.interface.utils."""

from typing import Any, cast

import numpy as np
import pytest
from gymnasium import spaces

from containerl.interface import proto_pb2, utils


def test_numpy_to_native_space_and_back_box() -> None:
    space = spaces.Box(
        low=np.array([-1.0, -2.0], dtype=np.float32),
        high=np.array([1.0, 2.0], dtype=np.float32),
        dtype=np.float32,
    )
    proto = proto_pb2.Space()
    utils.numpy_to_native_space(space, proto)

    assert proto.type == "Box"
    assert list(proto.low) == [-1.0, -2.0]
    assert list(proto.high) == [1.0, 2.0]
    assert list(proto.shape) == list(space.shape)

    reconstructed = utils.native_to_numpy_space(proto)
    assert isinstance(reconstructed, spaces.Box)
    assert reconstructed.shape == space.shape


def test_numpy_to_native_space_discrete_multidiscrete_multibinary() -> None:
    d = spaces.Discrete(5)
    p = proto_pb2.Space()
    utils.numpy_to_native_space(d, p)
    assert p.type == "Discrete" and p.n == 5

    md = spaces.MultiDiscrete([2, 3])
    p2 = proto_pb2.Space()
    utils.numpy_to_native_space(md, p2)
    assert p2.type == "MultiDiscrete" and list(p2.nvec) == [2, 3]

    mb = spaces.MultiBinary((4,))
    p3 = proto_pb2.Space()
    utils.numpy_to_native_space(mb, p3)
    assert p3.type == "MultiBinary" and list(p3.nvec) == [4]


def test_numpy_to_native_space_unsupported_raises() -> None:
    class Fake:
        pass

    with pytest.raises(ValueError):
        utils.numpy_to_native_space(cast(Any, Fake()), proto_pb2.Space())


def test_native_to_numpy_space_unsupported_raises() -> None:
    p = proto_pb2.Space()
    p.type = "Unknown"
    with pytest.raises(ValueError):
        utils.native_to_numpy_space(p)


def test_numpy_to_native_and_native_to_numpy_basic_types() -> None:
    assert utils.numpy_to_native(np.int32(3)) == 3
    arr = np.array([1, 2, 3])
    assert utils.numpy_to_native(arr) == [1, 2, 3]

    box = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    val = [0.1, 0.2, 0.3]
    out = utils.native_to_numpy(val, box)
    assert isinstance(out, np.ndarray) and out.shape == (3,)

    disc = spaces.Discrete(4)
    assert isinstance(utils.native_to_numpy(2, disc), np.integer)
    with pytest.raises(ValueError):
        utils.native_to_numpy([1], disc)

    md = spaces.MultiDiscrete([2, 2])
    assert isinstance(utils.native_to_numpy([0, 1], md), np.ndarray)

    mb = spaces.MultiBinary((2,))
    assert isinstance(utils.native_to_numpy([0, 1], mb), np.ndarray)


def test_native_to_numpy_vec_shapes() -> None:
    box = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    vec = [0.1, 0.2] * 3
    out = utils.native_to_numpy_vec(vec, box, num_envs=3)
    assert out.shape == (3, 2)

    disc = spaces.Discrete(3)
    out2 = utils.native_to_numpy_vec([0, 1, 2], disc, num_envs=3)
    assert out2.shape == (3,)

    md = spaces.MultiDiscrete([2, 2])
    out3 = utils.native_to_numpy_vec([0, 1, 0, 1, 1, 0], md, num_envs=3)
    assert out3.shape == (3, 2)


def test_native_to_numpy_vec_unsupported_raises() -> None:
    class Fake:
        pass

    with pytest.raises(ValueError):
        utils.native_to_numpy_vec([1], cast(Any, Fake()), 1)


def test_process_info_conversions() -> None:
    info = {
        "arr": np.array([1, 2]),
        "num": np.int32(5),
        "b": np.bool_(True),
        "lst": [np.int64(2), np.float32(3.1), np.bool_(False), np.array([1, 2])],
        "tup": (np.int64(4),),
    }

    processed = utils.process_info(info.copy())
    assert processed["arr"] == [1, 2]
    assert processed["num"] == 5
    assert processed["b"] is True
    # Compare elements individually to avoid float representation issues
    import math

    lst = cast(list[Any], processed["lst"])  # tell mypy this is a list
    assert lst[0] == 2
    assert math.isclose(float(lst[1]), 3.1, rel_tol=1e-6)
    assert lst[2] is False
    assert lst[3] == [1, 2]
    assert cast(list[Any], processed["tup"]) == [4]
