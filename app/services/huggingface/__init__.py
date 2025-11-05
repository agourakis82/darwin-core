"""
Darwin Hugging Face Integration

Complete integration for scientific publication:
- Model publishing (Darwin-Psych-Base, Darwin-Depression-Classifier, etc)
- Dataset publishing (Darwin-MentalHealth-BR, Darwin-EEG-Psych, etc)
- Benchmark suite (standardized evaluation)
- Model cards with comprehensive metadata
- Dataset cards with ethical considerations
- Gated access for medical data
"""

from .model_publisher import (
    HuggingFacePublisher,
    ModelCard,
    get_hf_publisher
)

from .dataset_publisher import (
    DatasetPublisher,
    DatasetCard,
    get_dataset_publisher
)

from .benchmark import (
    DarwinNeuroBenchmark,
    BenchmarkTask,
    BenchmarkMetrics,
    BenchmarkResult,
    get_benchmark
)

__all__ = [
    # Model Publishing
    'HuggingFacePublisher',
    'ModelCard',
    'get_hf_publisher',
    
    # Dataset Publishing
    'DatasetPublisher',
    'DatasetCard',
    'get_dataset_publisher',
    
    # Benchmarking
    'DarwinNeuroBenchmark',
    'BenchmarkTask',
    'BenchmarkMetrics',
    'BenchmarkResult',
    'get_benchmark',
]

__version__ = "1.0.0"

