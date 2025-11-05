import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class UserInteractionEvent(_message.Message):
    __slots__ = ("user_id", "platform", "domain", "query", "memory_saved", "search_performed", "timestamp", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    MEMORY_SAVED_FIELD_NUMBER: _ClassVar[int]
    SEARCH_PERFORMED_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    platform: str
    domain: str
    query: str
    memory_saved: bool
    search_performed: bool
    timestamp: _timestamp_pb2.Timestamp
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, user_id: _Optional[str] = ..., platform: _Optional[str] = ..., domain: _Optional[str] = ..., query: _Optional[str] = ..., memory_saved: bool = ..., search_performed: bool = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class MLTrainingEvent(_message.Message):
    __slots__ = ("model_name", "model_type", "status", "progress", "metrics", "timestamp")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    model_type: str
    status: str
    progress: float
    metrics: _containers.ScalarMap[str, float]
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, model_name: _Optional[str] = ..., model_type: _Optional[str] = ..., status: _Optional[str] = ..., progress: _Optional[float] = ..., metrics: _Optional[_Mapping[str, float]] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class PluginHealthEvent(_message.Message):
    __slots__ = ("plugin_name", "status", "details", "timestamp")
    class DetailsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PLUGIN_NAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    plugin_name: str
    status: str
    details: _containers.ScalarMap[str, str]
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, plugin_name: _Optional[str] = ..., status: _Optional[str] = ..., details: _Optional[_Mapping[str, str]] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class SystemAlertEvent(_message.Message):
    __slots__ = ("severity", "component", "message", "context", "timestamp")
    class ContextEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    COMPONENT_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    severity: str
    component: str
    message: str
    context: _containers.ScalarMap[str, str]
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, severity: _Optional[str] = ..., component: _Optional[str] = ..., message: _Optional[str] = ..., context: _Optional[_Mapping[str, str]] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ContinuousLearningEvent(_message.Message):
    __slots__ = ("event_type", "user_id", "new_preferences", "expertise_areas", "timestamp")
    class NewPreferencesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    NEW_PREFERENCES_FIELD_NUMBER: _ClassVar[int]
    EXPERTISE_AREAS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    event_type: str
    user_id: str
    new_preferences: _containers.ScalarMap[str, float]
    expertise_areas: _containers.RepeatedScalarFieldContainer[str]
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, event_type: _Optional[str] = ..., user_id: _Optional[str] = ..., new_preferences: _Optional[_Mapping[str, float]] = ..., expertise_areas: _Optional[_Iterable[str]] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...
