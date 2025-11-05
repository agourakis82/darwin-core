from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class KECResults(_message.Message):
    __slots__ = ("H_spectral", "k_forman_mean", "k_forman_p05", "k_forman_p95", "k_ollivier_mean", "k_ollivier_p05", "k_ollivier_p95", "sigma", "phi", "d_perc_um", "sigma_Q", "algorithm_version", "processing_time_ms", "random_seed", "segmentation_threshold")
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
    RANDOM_SEED_FIELD_NUMBER: _ClassVar[int]
    SEGMENTATION_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
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
    random_seed: str
    segmentation_threshold: float
    def __init__(self, H_spectral: _Optional[float] = ..., k_forman_mean: _Optional[float] = ..., k_forman_p05: _Optional[float] = ..., k_forman_p95: _Optional[float] = ..., k_ollivier_mean: _Optional[float] = ..., k_ollivier_p05: _Optional[float] = ..., k_ollivier_p95: _Optional[float] = ..., sigma: _Optional[float] = ..., phi: _Optional[float] = ..., d_perc_um: _Optional[float] = ..., sigma_Q: _Optional[float] = ..., algorithm_version: _Optional[str] = ..., processing_time_ms: _Optional[int] = ..., random_seed: _Optional[str] = ..., segmentation_threshold: _Optional[float] = ...) -> None: ...

class AnalyzeRequest(_message.Message):
    __slots__ = ("dataset_id", "options", "include_ollivier_ricci", "include_quantum_coherence")
    class OptionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_OLLIVIER_RICCI_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_QUANTUM_COHERENCE_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    options: _containers.ScalarMap[str, str]
    include_ollivier_ricci: bool
    include_quantum_coherence: bool
    def __init__(self, dataset_id: _Optional[str] = ..., options: _Optional[_Mapping[str, str]] = ..., include_ollivier_ricci: bool = ..., include_quantum_coherence: bool = ...) -> None: ...

class ScaffoldAnalysisResponse(_message.Message):
    __slots__ = ("success", "kec", "meta", "error_message")
    class MetaEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    KEC_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    kec: KECResults
    meta: _containers.ScalarMap[str, str]
    error_message: str
    def __init__(self, success: bool = ..., kec: _Optional[_Union[KECResults, _Mapping]] = ..., meta: _Optional[_Mapping[str, str]] = ..., error_message: _Optional[str] = ...) -> None: ...

class DataChunk(_message.Message):
    __slots__ = ("data", "chunk_index", "total_chunks", "dataset_id")
    DATA_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    chunk_index: int
    total_chunks: int
    dataset_id: str
    def __init__(self, data: _Optional[bytes] = ..., chunk_index: _Optional[int] = ..., total_chunks: _Optional[int] = ..., dataset_id: _Optional[str] = ...) -> None: ...

class ProcessedChunk(_message.Message):
    __slots__ = ("chunk_index", "progress_percent", "status_message")
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_PERCENT_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    chunk_index: int
    progress_percent: float
    status_message: str
    def __init__(self, chunk_index: _Optional[int] = ..., progress_percent: _Optional[float] = ..., status_message: _Optional[str] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ("plugin_id",)
    PLUGIN_ID_FIELD_NUMBER: _ClassVar[int]
    plugin_id: str
    def __init__(self, plugin_id: _Optional[str] = ...) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("healthy", "status_message", "metrics")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    status_message: str
    metrics: _containers.ScalarMap[str, str]
    def __init__(self, healthy: bool = ..., status_message: _Optional[str] = ..., metrics: _Optional[_Mapping[str, str]] = ...) -> None: ...
