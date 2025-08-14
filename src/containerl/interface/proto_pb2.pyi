from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EnvironmentType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STANDARD: _ClassVar[EnvironmentType]
    VECTORIZED: _ClassVar[EnvironmentType]
STANDARD: EnvironmentType
VECTORIZED: EnvironmentType

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ObservationRequest(_message.Message):
    __slots__ = ("observation",)
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    observation: bytes
    def __init__(self, observation: _Optional[bytes] = ...) -> None: ...

class ActionResponse(_message.Message):
    __slots__ = ("action",)
    ACTION_FIELD_NUMBER: _ClassVar[int]
    action: bytes
    def __init__(self, action: _Optional[bytes] = ...) -> None: ...

class InitRequest(_message.Message):
    __slots__ = ("render_mode", "init_args")
    RENDER_MODE_FIELD_NUMBER: _ClassVar[int]
    INIT_ARGS_FIELD_NUMBER: _ClassVar[int]
    render_mode: str
    init_args: bytes
    def __init__(self, render_mode: _Optional[str] = ..., init_args: _Optional[bytes] = ...) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ("seed", "options")
    SEED_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    seed: int
    options: bytes
    def __init__(self, seed: _Optional[int] = ..., options: _Optional[bytes] = ...) -> None: ...

class StepRequest(_message.Message):
    __slots__ = ("action",)
    ACTION_FIELD_NUMBER: _ClassVar[int]
    action: bytes
    def __init__(self, action: _Optional[bytes] = ...) -> None: ...

class ResetResponse(_message.Message):
    __slots__ = ("observation", "info")
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    observation: bytes
    info: bytes
    def __init__(self, observation: _Optional[bytes] = ..., info: _Optional[bytes] = ...) -> None: ...

class StepResponse(_message.Message):
    __slots__ = ("observation", "reward", "terminated", "truncated", "info")
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    TERMINATED_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    observation: bytes
    reward: bytes
    terminated: bytes
    truncated: bytes
    info: bytes
    def __init__(self, observation: _Optional[bytes] = ..., reward: _Optional[bytes] = ..., terminated: _Optional[bytes] = ..., truncated: _Optional[bytes] = ..., info: _Optional[bytes] = ...) -> None: ...

class RenderResponse(_message.Message):
    __slots__ = ("render_data",)
    RENDER_DATA_FIELD_NUMBER: _ClassVar[int]
    render_data: bytes
    def __init__(self, render_data: _Optional[bytes] = ...) -> None: ...

class Space(_message.Message):
    __slots__ = ("type", "low", "high", "n", "nvec", "shape", "dtype")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    LOW_FIELD_NUMBER: _ClassVar[int]
    HIGH_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    NVEC_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    type: str
    low: _containers.RepeatedScalarFieldContainer[float]
    high: _containers.RepeatedScalarFieldContainer[float]
    n: int
    nvec: _containers.RepeatedScalarFieldContainer[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    def __init__(self, type: _Optional[str] = ..., low: _Optional[_Iterable[float]] = ..., high: _Optional[_Iterable[float]] = ..., n: _Optional[int] = ..., nvec: _Optional[_Iterable[int]] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ...) -> None: ...

class SpacesResponse(_message.Message):
    __slots__ = ("observation_space", "action_space", "num_envs", "environment_type", "render_mode")
    class ObservationSpaceEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Space
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Space, _Mapping]] = ...) -> None: ...
    OBSERVATION_SPACE_FIELD_NUMBER: _ClassVar[int]
    ACTION_SPACE_FIELD_NUMBER: _ClassVar[int]
    NUM_ENVS_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    RENDER_MODE_FIELD_NUMBER: _ClassVar[int]
    observation_space: _containers.MessageMap[str, Space]
    action_space: Space
    num_envs: int
    environment_type: EnvironmentType
    render_mode: str
    def __init__(self, observation_space: _Optional[_Mapping[str, Space]] = ..., action_space: _Optional[_Union[Space, _Mapping]] = ..., num_envs: _Optional[int] = ..., environment_type: _Optional[_Union[EnvironmentType, str]] = ..., render_mode: _Optional[str] = ...) -> None: ...
