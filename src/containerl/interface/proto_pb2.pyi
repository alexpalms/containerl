from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

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
    def __init__(self, observation: bytes | None = ...) -> None: ...

class ActionResponse(_message.Message):
    __slots__ = ("action",)
    ACTION_FIELD_NUMBER: _ClassVar[int]
    action: bytes
    def __init__(self, action: bytes | None = ...) -> None: ...

class AgentInitRequest(_message.Message):
    __slots__ = ("init_args",)
    INIT_ARGS_FIELD_NUMBER: _ClassVar[int]
    init_args: bytes
    def __init__(self, init_args: bytes | None = ...) -> None: ...

class EnvInitRequest(_message.Message):
    __slots__ = ("init_args",)
    INIT_ARGS_FIELD_NUMBER: _ClassVar[int]
    init_args: bytes
    def __init__(self, init_args: bytes | None = ...) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ("seed", "options")
    SEED_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    seed: int
    options: bytes
    def __init__(self, seed: int | None = ..., options: bytes | None = ...) -> None: ...

class StepRequest(_message.Message):
    __slots__ = ("action",)
    ACTION_FIELD_NUMBER: _ClassVar[int]
    action: bytes
    def __init__(self, action: bytes | None = ...) -> None: ...

class ResetResponse(_message.Message):
    __slots__ = ("observation", "info")
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    observation: bytes
    info: bytes
    def __init__(
        self, observation: bytes | None = ..., info: bytes | None = ...
    ) -> None: ...

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
    def __init__(
        self,
        observation: bytes | None = ...,
        reward: bytes | None = ...,
        terminated: bytes | None = ...,
        truncated: bytes | None = ...,
        info: bytes | None = ...,
    ) -> None: ...

class RenderResponse(_message.Message):
    __slots__ = ("render_data",)
    RENDER_DATA_FIELD_NUMBER: _ClassVar[int]
    render_data: bytes
    def __init__(self, render_data: bytes | None = ...) -> None: ...

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
    def __init__(
        self,
        type: str | None = ...,
        low: _Iterable[float] | None = ...,
        high: _Iterable[float] | None = ...,
        n: int | None = ...,
        nvec: _Iterable[int] | None = ...,
        shape: _Iterable[int] | None = ...,
        dtype: str | None = ...,
    ) -> None: ...

class EnvInitResponse(_message.Message):
    __slots__ = (
        "observation_space",
        "action_space",
        "num_envs",
        "environment_type",
        "render_mode",
        "info",
    )
    class ObservationSpaceEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Space
        def __init__(
            self, key: str | None = ..., value: Space | None = ...
        ) -> None: ...

    OBSERVATION_SPACE_FIELD_NUMBER: _ClassVar[int]
    ACTION_SPACE_FIELD_NUMBER: _ClassVar[int]
    NUM_ENVS_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    RENDER_MODE_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    observation_space: _containers.MessageMap[str, Space]
    action_space: Space
    num_envs: int
    environment_type: EnvironmentType
    render_mode: str
    info: bytes
    def __init__(
        self,
        observation_space: _Mapping[str, Space] | None = ...,
        action_space: Space | None = ...,
        num_envs: int | None = ...,
        environment_type: EnvironmentType | str | None = ...,
        render_mode: str | None = ...,
        info: bytes | None = ...,
    ) -> None: ...

class AgentInitResponse(_message.Message):
    __slots__ = ("observation_space", "action_space", "info")
    class ObservationSpaceEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Space
        def __init__(
            self, key: str | None = ..., value: Space | None = ...
        ) -> None: ...

    OBSERVATION_SPACE_FIELD_NUMBER: _ClassVar[int]
    ACTION_SPACE_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    observation_space: _containers.MessageMap[str, Space]
    action_space: Space
    info: bytes
    def __init__(
        self,
        observation_space: _Mapping[str, Space] | None = ...,
        action_space: Space | None = ...,
        info: bytes | None = ...,
    ) -> None: ...
