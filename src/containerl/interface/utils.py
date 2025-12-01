"""Utility functions for converting between Gymnasium spaces and protobuf representations."""

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .proto_pb2 import Space

AllowedTypes = NDArray[np.floating | np.integer] | np.integer
AllowedSerializableTypes = list[int | float] | int
AllowedSpaces = spaces.Space[
    AllowedTypes
]  # spaces.Box | spaces.Discrete | spaces.MultiDiscrete | spaces.MultiBinary
AllowedInfoBaseTypes = str | bool | int | float
AllowedInfoValueTypes = AllowedInfoBaseTypes | list[AllowedInfoBaseTypes]


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
    elif isinstance(space, spaces.MultiBinary):
        space_proto.type = "MultiBinary"
        space_proto.nvec.extend(list(space.shape))
        space_proto.shape.extend(space.shape)
        space_proto.dtype = str(space.dtype)
    else:
        raise ValueError(f"Unsupported space type: {type(space)}")


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


def numpy_to_native(obj: AllowedTypes) -> AllowedSerializableTypes:
    """Convert numpy arrays and other non-serializable objects to serializable types based on the space.

    Args:
        obj: The object to convert
        space: The Gymnasium space object (Box, Discrete, MultiDiscrete, or MultiBinary)
    """
    # Handle the four base space types
    if isinstance(obj, type(np.integer)):
        return int(obj)
    else:
        return obj.tolist()


def native_to_numpy(
    obj: AllowedSerializableTypes, space: AllowedSpaces
) -> AllowedTypes:
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
    elif isinstance(space, spaces.MultiBinary):
        return np.array(obj, dtype=np.int8).reshape(space.shape)
    else:
        raise ValueError(f"Unsupported space type: {type(space)}")


# TODO: address when dealing with vectorized envs properly
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
        return np.array(obj, dtype=np.int64).reshape(num_envs, *())
    elif isinstance(space, spaces.MultiDiscrete):
        return np.array(obj, dtype=np.int64).reshape(num_envs, *space.shape)
    elif isinstance(space, spaces.MultiBinary):
        return np.array(obj, dtype=np.int8).reshape(num_envs, *space.shape)
    else:
        raise ValueError(f"Unsupported space type: {type(space)}")


def process_info(
    info: dict[
        str,
        AllowedInfoBaseTypes
        | NDArray[np.floating | np.integer]
        | np.floating
        | np.integer
        | np.bool_
        | list[AllowedInfoBaseTypes],
    ],
) -> dict[str, AllowedInfoValueTypes]:
    """Process the info dictionary to convert numpy types to native Python types."""
    processed_info: dict[str, AllowedInfoValueTypes] = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            processed_info[key] = value.tolist()
        elif isinstance(value, np.number):  # Catches all numeric types (int, float)
            processed_info[key] = value.item()  # .item() converts to native Python type
        elif isinstance(value, np.bool_):
            processed_info[key] = bool(value)
        elif (
            isinstance(value, str)
            | isinstance(value, bool)
            | isinstance(value, int)
            | isinstance(value, float)
        ):
            processed_info[key] = value
        elif isinstance(value, list):
            # Process lists to convert any numpy types within
            processed_list: list[AllowedInfoBaseTypes] = []
            for item in value:
                if (
                    isinstance(item, str)
                    | isinstance(item, bool)
                    | isinstance(item, int)
                    | isinstance(item, float)
                ):
                    processed_list.append(item)

            processed_info[key] = processed_list

    return processed_info
