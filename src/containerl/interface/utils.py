"""Utility functions for converting between Gymnasium spaces and protobuf representations."""

import json
from typing import cast

import numpy as np
from google.protobuf import json_format
from gymnasium import spaces

from containerl.interface.proto_pb2 import EnvironmentType, Space

AllowedSpaces = spaces.Box | spaces.Discrete | spaces.MultiDiscrete | spaces.MultiBinary


def numpy_to_native_space(space: AllowedSpaces, space_proto: Space) -> None:
    """Set space information based on space type."""
    if isinstance(space, spaces.Box):
        space_proto.type = "Box"
        space_proto.low.extend(space.low.flatten().tolist())
        space_proto.high.extend(space.high.flatten().tolist())
        space_proto.shape.extend(space.shape)
        space_proto.dtype = str(space.dtype)
    elif isinstance(space, spaces.Discrete):
        space_proto.type = "Discrete"
        space_proto.n = space.n.item()
        space_proto.shape.extend([1])  # Discrete spaces have shape (1,)
        space_proto.dtype = str(space.dtype)
    elif isinstance(space, spaces.MultiDiscrete):
        space_proto.type = "MultiDiscrete"
        space_proto.nvec.extend(space.nvec.tolist())
        space_proto.shape.extend(space.shape)
        space_proto.dtype = str(space.dtype)
    else:  # MultiBinary
        space_proto.type = "MultiBinary"
        space_proto.nvec.extend(list(space.shape))
        space_proto.shape.extend(space.shape)
        space_proto.dtype = str(space.dtype)


def native_to_numpy_space(proto_space: Space) -> AllowedSpaces:
    """Create a Gym action space from the protobuf space definition."""
    if proto_space.type == "Box":
        # Create a Box space
        low = np.array(proto_space.low, dtype=np.float32)
        high = np.array(proto_space.high, dtype=np.float32)
        shape = tuple(proto_space.shape)
        # Reshape the low and high arrays
        low = low.reshape(shape)
        high = high.reshape(shape)
        return spaces.Box(low=low, high=high, dtype=np.float32)
    elif proto_space.type == "Discrete":
        # Create a Discrete space with n possible actions
        return spaces.Discrete(proto_space.n)
    elif proto_space.type == "MultiDiscrete":
        # Create a MultiDiscrete space
        nvec = np.array(proto_space.nvec, dtype=np.int64)
        return spaces.MultiDiscrete(nvec)
    elif proto_space.type == "MultiBinary":
        # Create a MultiBinary space
        n = proto_space.nvec
        return spaces.MultiBinary(n[0] if len(n) == 1 else n)
    else:
        raise ValueError(f"Unsupported space type: {proto_space.type}")


def space_proto_to_json(space_proto: Space) -> dict[str, str | int | list[float | int]]:
    """Convert a Space protobuf message to JSON dictionary."""
    json_str = json_format.MessageToJson(space_proto)
    json_dict: dict[str, str | int | list[float | int]] = json.loads(json_str)
    return json_dict


def json_to_space_proto(json_dict: dict[str, str | int | list[float | int]]) -> Space:
    """Convert JSON dictionary back to a Space protobuf message."""
    json_str = json.dumps(json_dict)
    space_proto = Space()
    json_format.Parse(json_str, space_proto)
    return space_proto


def generate_spaces_info_from_gym_spaces(
    observation_space: dict[str, AllowedSpaces],
    action_space: AllowedSpaces,
    environment_type: EnvironmentType | None = None,
) -> dict[
    str,
    dict[str, dict[str, str | int | list[float | int]]]
    | dict[str, str | int | list[float | int]]
    | EnvironmentType
    | None,
]:
    """Generate space information dictionaries from Gymnasium spaces."""
    observation_space_info: dict[str, dict[str, str | int | list[float | int]]] = {}
    for key in sorted(observation_space.keys()):
        value = observation_space[key]
        proto_value = Space()
        numpy_to_native_space(value, proto_value)
        json_value_obs = space_proto_to_json(proto_value)
        observation_space_info[key] = json_value_obs

    proto_value = Space()
    numpy_to_native_space(action_space, proto_value)
    json_value = space_proto_to_json(proto_value)
    action_space_info = json_value

    results: dict[
        str,
        dict[str, dict[str, str | int | list[float | int]]]
        | dict[str, str | int | list[float | int]]
        | EnvironmentType
        | None,
    ] = {
        "observationSpaceInfo": observation_space_info,
        "actionSpaceInfo": action_space_info,
        "environmentType": environment_type,
    }

    return results


def numpy_to_native(
    obj: np.ndarray, space: AllowedSpaces
) -> list[int | float] | int | float:
    """Convert numpy arrays and other non-serializable objects to serializable types based on the space.

    Args:
        obj: The object to convert
        space: The Gymnasium space object (Box, Discrete, MultiDiscrete, or MultiBinary)
    """
    # Handle the four base space types
    if isinstance(space, spaces.Discrete):
        return cast(int | float, obj.item())
    else:
        return cast(list[int | float], obj.tolist())


def native_to_numpy(
    obj: list[int | float] | int | float, space: AllowedSpaces
) -> np.ndarray | np.int64:
    """Convert serialized objects back to their original form based on space.

    Args:
        obj: The object to convert
        space: The Gymnasium space object (Box, Discrete, MultiDiscrete, or MultiBinary)
    """
    if isinstance(space, spaces.Box):
        return np.array(obj, dtype=space.dtype).reshape(space.shape)
    elif isinstance(space, spaces.Discrete):
        if isinstance(obj, int):
            return np.int64(obj)
        else:
            raise ValueError("Expected int for Discrete space deserialization")
    elif isinstance(space, spaces.MultiDiscrete):
        return np.array(obj, dtype=np.int64).reshape(space.shape)
    else:  # MultiBinary
        return np.array(obj, dtype=np.int8).reshape(space.shape)


def native_to_numpy_vec(
    obj: list[int | float], space: AllowedSpaces, num_envs: int
) -> np.ndarray:
    """Convert serialized objects back to their original form based on space.

    Args:
        obj: The object to convert
        space: The Gymnasium space object (Box, Discrete, MultiDiscrete, or MultiBinary)
        num_envs: The number of environments
    """
    if isinstance(space, spaces.Box):
        return np.array(obj, dtype=space.dtype).reshape(num_envs, *space.shape)
    elif isinstance(space, spaces.Discrete):
        return np.array(obj, dtype=np.int64).reshape(num_envs, 1)
    elif isinstance(space, spaces.MultiDiscrete):
        return np.array(obj, dtype=np.int64).reshape(num_envs, *space.shape)
    else:  # MultiBinary
        return np.array(obj, dtype=np.int8).reshape(num_envs, *space.shape)
