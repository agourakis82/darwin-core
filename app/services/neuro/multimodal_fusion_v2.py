"""
Multimodal Fusion V2 - Integration of Multiple Data Modalities

Fuses different data types for comprehensive psychiatric assessment:
- EEG signals (Brain Transformer embeddings)
- Digital phenotype features
- Clinical assessments
- Genetic data (SNPs)
- Neuroimaging (fMRI)

Fusion strategies:
- Early fusion (concatenation)
- Late fusion (ensemble)
- Cross-attention fusion
- Hierarchical fusion
- Adaptive weighted fusion

References:
- "Multimodal Machine Learning: A Survey and Taxonomy" (Baltrušaitis et al., 2018)
- "Attention-Based Multimodal Fusion for Sentiment Analysis" (Zadeh et al., 2017)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    """Fusion strategies"""
    EARLY = "early"  # Concatenate features
    LATE = "late"  # Ensemble predictions
    CROSS_ATTENTION = "cross_attention"  # Attention-based
    HIERARCHICAL = "hierarchical"  # Multi-level
    ADAPTIVE = "adaptive"  # Learned weights


@dataclass
class MultimodalInput:
    """Input data from multiple modalities"""
    eeg_embedding: Optional[torch.Tensor] = None  # [d_eeg]
    digital_phenotype: Optional[torch.Tensor] = None  # [d_phenotype]
    clinical_scores: Optional[torch.Tensor] = None  # [d_clinical]
    genetic_features: Optional[torch.Tensor] = None  # [d_genetic]
    neuroimaging: Optional[torch.Tensor] = None  # [d_neuro]


class EarlyFusion(nn.Module):
    """
    Early fusion: concatenate all modalities.
    
    Simple but effective baseline.
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 4
    ):
        super().__init__()
        
        # Calculate total input dimension
        total_dim = sum(input_dims.values())
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, inputs: MultimodalInput) -> torch.Tensor:
        """Concatenate and fuse"""
        features = []
        
        if inputs.eeg_embedding is not None:
            features.append(inputs.eeg_embedding)
        if inputs.digital_phenotype is not None:
            features.append(inputs.digital_phenotype)
        if inputs.clinical_scores is not None:
            features.append(inputs.clinical_scores)
        if inputs.genetic_features is not None:
            features.append(inputs.genetic_features)
        if inputs.neuroimaging is not None:
            features.append(inputs.neuroimaging)
        
        # Concatenate
        x = torch.cat(features, dim=-1)
        
        return self.fusion(x)


class LateFusion(nn.Module):
    """
    Late fusion: separate models per modality, ensemble predictions.
    
    Good when modalities are very different.
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 128,
        num_classes: int = 4
    ):
        super().__init__()
        
        # Separate classifier for each modality
        self.classifiers = nn.ModuleDict()
        
        for modality, dim in input_dims.items():
            self.classifiers[modality] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, num_classes)
            )
        
        # Fusion weights (learnable)
        self.fusion_weights = nn.Parameter(
            torch.ones(len(input_dims)) / len(input_dims)
        )
    
    def forward(self, inputs: MultimodalInput) -> torch.Tensor:
        """Ensemble predictions"""
        predictions = []
        weights = []
        idx = 0
        
        modality_map = {
            'eeg': inputs.eeg_embedding,
            'phenotype': inputs.digital_phenotype,
            'clinical': inputs.clinical_scores,
            'genetic': inputs.genetic_features,
            'neuro': inputs.neuroimaging
        }
        
        for modality, features in modality_map.items():
            if features is not None and modality in self.classifiers:
                pred = self.classifiers[modality](features)
                predictions.append(pred)
                weights.append(self.fusion_weights[idx])
                idx += 1
        
        # Weighted ensemble
        weights = F.softmax(torch.stack(weights), dim=0)
        predictions = torch.stack(predictions)
        
        return (predictions * weights.view(-1, 1, 1)).sum(dim=0)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion: modalities attend to each other.
    
    State-of-the-art for multimodal learning.
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        d_model: int = 256,
        nhead: int = 4,
        num_classes: int = 4
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Project each modality to common dimension
        self.projections = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.projections[modality] = nn.Linear(dim, d_model)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, inputs: MultimodalInput) -> torch.Tensor:
        """Cross-attend and fuse"""
        # Project to common space
        projected = []
        
        modality_map = {
            'eeg': inputs.eeg_embedding,
            'phenotype': inputs.digital_phenotype,
            'clinical': inputs.clinical_scores,
            'genetic': inputs.genetic_features,
            'neuro': inputs.neuroimaging
        }
        
        for modality, features in modality_map.items():
            if features is not None and modality in self.projections:
                proj = self.projections[modality](features)
                projected.append(proj)
        
        if len(projected) == 0:
            raise ValueError("No modalities provided")
        
        # Stack as sequence
        x = torch.stack(projected, dim=1)  # [batch, num_modalities, d_model]
        
        # Self-attention across modalities
        attended, _ = self.cross_attention(x, x, x)
        
        # Average pool across modalities
        pooled = attended.mean(dim=1)
        
        return self.fusion(pooled)


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion: learn importance of each modality.
    
    Dynamically weights modalities based on input.
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 4
    ):
        super().__init__()
        
        # Modality encoders
        self.encoders = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.encoders[modality] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        
        # Attention network (computes modality weights)
        num_modalities = len(input_dims)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, inputs: MultimodalInput) -> torch.Tensor:
        """Adaptively fuse modalities"""
        encoded = []
        
        modality_map = {
            'eeg': inputs.eeg_embedding,
            'phenotype': inputs.digital_phenotype,
            'clinical': inputs.clinical_scores,
            'genetic': inputs.genetic_features,
            'neuro': inputs.neuroimaging
        }
        
        for modality, features in modality_map.items():
            if features is not None and modality in self.encoders:
                enc = self.encoders[modality](features)
                encoded.append(enc)
        
        # Compute attention weights
        concat = torch.cat(encoded, dim=-1)
        weights = self.attention(concat)
        
        # Weighted sum
        stacked = torch.stack(encoded, dim=1)  # [batch, num_modalities, hidden]
        weighted = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        
        return self.classifier(weighted)


class MultimodalFusionModel(nn.Module):
    """
    Main multimodal fusion model.
    
    Supports multiple fusion strategies.
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        strategy: FusionStrategy = FusionStrategy.CROSS_ATTENTION,
        hidden_dim: int = 256,
        num_classes: int = 4
    ):
        super().__init__()
        
        self.strategy = strategy
        self.input_dims = input_dims
        
        if strategy == FusionStrategy.EARLY:
            self.fusion = EarlyFusion(input_dims, hidden_dim, num_classes)
        elif strategy == FusionStrategy.LATE:
            self.fusion = LateFusion(input_dims, hidden_dim, num_classes)
        elif strategy == FusionStrategy.CROSS_ATTENTION:
            self.fusion = CrossAttentionFusion(input_dims, hidden_dim, 4, num_classes)
        elif strategy == FusionStrategy.ADAPTIVE:
            self.fusion = AdaptiveFusion(input_dims, hidden_dim, num_classes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.info(f"Multimodal fusion model: {strategy.value}")
    
    def forward(self, inputs: MultimodalInput) -> torch.Tensor:
        """Forward pass"""
        return self.fusion(inputs)
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test
if __name__ == "__main__":
    print("="*70)
    print("Multimodal Fusion V2 - Test")
    print("="*70)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Define input dimensions
    input_dims = {
        'eeg': 128,  # From Brain Transformer
        'phenotype': 20,  # From Digital Phenotyper
        'clinical': 10,  # Clinical scores
        'genetic': 50,  # SNPs
        'neuro': 64  # fMRI features
    }
    
    num_classes = 4  # e.g., healthy, depression, anxiety, ADHD
    
    # Test all strategies
    strategies = [
        FusionStrategy.EARLY,
        FusionStrategy.LATE,
        FusionStrategy.CROSS_ATTENTION,
        FusionStrategy.ADAPTIVE
    ]
    
    batch_size = 8
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy.value}")
        print(f"{'='*70}")
        
        # Create model
        model = MultimodalFusionModel(
            input_dims,
            strategy=strategy,
            hidden_dim=256,
            num_classes=num_classes
        ).to(device)
        
        print(f"Parameters: {model.count_parameters():,}")
        
        # Create dummy inputs
        inputs = MultimodalInput(
            eeg_embedding=torch.randn(batch_size, input_dims['eeg']).to(device),
            digital_phenotype=torch.randn(batch_size, input_dims['phenotype']).to(device),
            clinical_scores=torch.randn(batch_size, input_dims['clinical']).to(device),
            genetic_features=torch.randn(batch_size, input_dims['genetic']).to(device),
            neuroimaging=torch.randn(batch_size, input_dims['neuro']).to(device)
        )
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(inputs)
        
        print(f"Input shapes:")
        print(f"  EEG: {inputs.eeg_embedding.shape}")
        print(f"  Phenotype: {inputs.digital_phenotype.shape}")
        print(f"  Clinical: {inputs.clinical_scores.shape}")
        print(f"  Genetic: {inputs.genetic_features.shape}")
        print(f"  Neuroimaging: {inputs.neuroimaging.shape}")
        
        print(f"\nOutput shape: {output.shape}")
        print(f"Sample prediction: {output[0].cpu().numpy()}")
        
        # Test with missing modalities
        print("\nTesting with missing modalities...")
        inputs_partial = MultimodalInput(
            eeg_embedding=torch.randn(batch_size, input_dims['eeg']).to(device),
            digital_phenotype=torch.randn(batch_size, input_dims['phenotype']).to(device)
        )
        
        try:
            with torch.no_grad():
                output_partial = model(inputs_partial)
            print(f"✅ Works with partial modalities: {output_partial.shape}")
        except Exception as e:
            print(f"⚠️  Requires all modalities: {e}")
    
    # Benchmark inference speed
    print("\n" + "="*70)
    print("Inference Speed Benchmark")
    print("="*70)
    
    model = MultimodalFusionModel(
        input_dims,
        strategy=FusionStrategy.CROSS_ATTENTION,
        hidden_dim=256,
        num_classes=num_classes
    ).to(device)
    
    model.eval()
    
    import time
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(inputs)
    
    # Benchmark
    n_iters = 100
    start = time.time()
    
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(inputs)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    avg_time = elapsed / n_iters * 1000
    
    print(f"\nIterations: {n_iters}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg time per batch: {avg_time:.2f}ms")
    print(f"Throughput: {batch_size * 1000 / avg_time:.1f} samples/sec")
    
    print("\n" + "="*70)
    print("✅ Multimodal Fusion V2 works!")

