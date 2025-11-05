from google.protobuf import any_pb2 as _any_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MetadataRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class PluginMetadata(_message.Message):
    __slots__ = ("name", "version", "gpu_required", "gpu_memory_mb", "capabilities", "proto_version", "requirements")
    class RequirementsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    GPU_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    PROTO_VERSION_FIELD_NUMBER: _ClassVar[int]
    REQUIREMENTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    gpu_required: bool
    gpu_memory_mb: int
    capabilities: _containers.RepeatedScalarFieldContainer[str]
    proto_version: str
    requirements: _containers.ScalarMap[str, str]
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., gpu_required: bool = ..., gpu_memory_mb: _Optional[int] = ..., capabilities: _Optional[_Iterable[str]] = ..., proto_version: _Optional[str] = ..., requirements: _Optional[_Mapping[str, str]] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("healthy", "status", "details")
    class DetailsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    status: str
    details: _containers.ScalarMap[str, str]
    def __init__(self, healthy: bool = ..., status: _Optional[str] = ..., details: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ExecuteRequest(_message.Message):
    __slots__ = ("operation", "payload", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    operation: str
    payload: _any_pb2.Any
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, operation: _Optional[str] = ..., payload: _Optional[_Union[_any_pb2.Any, _Mapping]] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ExecuteResponse(_message.Message):
    __slots__ = ("success", "result", "metrics", "error")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    result: _any_pb2.Any
    metrics: _containers.ScalarMap[str, str]
    error: str
    def __init__(self, success: bool = ..., result: _Optional[_Union[_any_pb2.Any, _Mapping]] = ..., metrics: _Optional[_Mapping[str, str]] = ..., error: _Optional[str] = ...) -> None: ...

class DataChunk(_message.Message):
    __slots__ = ("data", "sequence", "is_last", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DATA_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    IS_LAST_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    sequence: int
    is_last: bool
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, data: _Optional[bytes] = ..., sequence: _Optional[int] = ..., is_last: bool = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...
