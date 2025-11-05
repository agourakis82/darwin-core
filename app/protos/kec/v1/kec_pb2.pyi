from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class KECResults(_message.Message):
    __slots__ = ("H_spectral", "k_forman_mean", "k_forman_p05", "k_forman_p95", "k_ollivier_mean", "k_ollivier_p05", "k_ollivier_p95", "sigma", "phi", "d_perc_um", "sigma_Q", "algorithm_version", "processing_time_ms", "metrics")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    H_SPECTRAL_FIELD_NUMBER: _ClassVar[int]
    K_FORMAN_MEAN_FIELD_NUMBER: _ClassVar[int]
    K_FORMAN_P05_FIELD_NUMBER: _ClassVar[int]
    K_FORMAN_P95_FIELD_NUMBER: _ClassVar[int]
    K_OLLIVIER_MEAN_FIELD_NUMBER: _ClassVar[int]
    K_OLLIVIER_P05_FIELD_NUMBER: _ClassVar[int]
    K_OLLIVIER_P95_FIELD_NUMBER: _ClassVar[int]
    SIGMA_FIELD_NUMBER: _ClassVar[int]
    PHI_FIELD_NUMBER: _ClassVar[int]
    D_PERC_UM_FIELD_NUMBER: _ClassVar[int]
    SIGMA_Q_FIELD_NUMBER: _ClassVar[int]
    ALGORITHM_VERSION_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    H_spectral: float
    k_forman_mean: float
    k_forman_p05: float
    k_forman_p95: float
    k_ollivier_mean: float
    k_ollivier_p05: float
    k_ollivier_p95: float
    sigma: float
    phi: float
    d_perc_um: float
    sigma_Q: float
    algorithm_version: str
    processing_time_ms: int
    metrics: _containers.ScalarMap[str, str]
    def __init__(self, H_spectral: _Optional[float] = ..., k_forman_mean: _Optional[float] = ..., k_forman_p05: _Optional[float] = ..., k_forman_p95: _Optional[float] = ..., k_ollivier_mean: _Optional[float] = ..., k_ollivier_p05: _Optional[float] = ..., k_ollivier_p95: _Optional[float] = ..., sigma: _Optional[float] = ..., phi: _Optional[float] = ..., d_perc_um: _Optional[float] = ..., sigma_Q: _Optional[float] = ..., algorithm_version: _Optional[str] = ..., processing_time_ms: _Optional[int] = ..., metrics: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ScaffoldAnalysisRequest(_message.Message):
    __slots__ = ("dataset_id", "include_ollivier_ricci", "include_quantum_coherence", "options")
    class OptionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_OLLIVIER_RICCI_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_QUANTUM_COHERENCE_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    include_ollivier_ricci: bool
    include_quantum_coherence: bool
    options: _containers.ScalarMap[str, str]
    def __init__(self, dataset_id: _Optional[str] = ..., include_ollivier_ricci: bool = ..., include_quantum_coherence: bool = ..., options: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ScaffoldAnalysisResponse(_message.Message):
    __slots__ = ("success", "kec", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    KEC_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    kec: KECResults
    error_message: str
    def __init__(self, success: bool = ..., kec: _Optional[_Union[KECResults, _Mapping]] = ..., error_message: _Optional[str] = ...) -> None: ...

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
