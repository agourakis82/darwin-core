"""
Darwin Neuro: Computational Neuroscience & Psychiatry Stack

Complete integration of:
- EEG/fMRI signal processing
- Brain transformers (BrainBERT)
- Digital phenotyping
- Multimodal fusion
- Clinical biomarkers

Clinical applications:
- Depression diagnosis and monitoring
- Bipolar disorder episode prediction
- Schizophrenia relapse prevention
- ADHD assessment
- Anxiety disorder profiling
- Treatment response prediction
"""

from .eeg_processor import (
    EEGProcessor,
    EEGConfig,
    EEGBand,
    get_eeg_processor
)

from .brain_transformer import (
    BrainTransformer,
    BrainTransformerConfig,
    BrainTransformerTrainer,
    get_brain_transformer,
    pretrain_brain_transformer
)

from .digital_phenotyping import (
    DigitalPhenotyping,
    DigitalPhenotype,
    MobilityMetrics,
    SocialMetrics,
    PhoneUsageMetrics,
    SleepMetrics,
    DataStream
)

from .multimodal_fusion import (
    MultimodalFusion,
    MultimodalSample,
    ModalityData,
    Modality,
    FusionStrategy
)

__all__ = [
    # EEG Processing
    'EEGProcessor',
    'EEGConfig',
    'EEGBand',
    'get_eeg_processor',
    
    # Brain Transformers
    'BrainTransformer',
    'BrainTransformerConfig',
    'BrainTransformerTrainer',
    'get_brain_transformer',
    'pretrain_brain_transformer',
    
    # Digital Phenotyping
    'DigitalPhenotyping',
    'DigitalPhenotype',
    'MobilityMetrics',
    'SocialMetrics',
    'PhoneUsageMetrics',
    'SleepMetrics',
    'DataStream',
    
    # Multimodal Fusion
    'MultimodalFusion',
    'MultimodalSample',
    'ModalityData',
    'Modality',
    'FusionStrategy',
]

__version__ = "1.0.0"

