"""
Multimodal Fusion for Computational Psychiatry

Integrates multiple data modalities for comprehensive psychiatric assessment:
- Neuroimaging: EEG, fMRI, MRI (structural)
- Genetics: SNPs, polygenic risk scores
- Clinical: symptoms, history, medications
- Behavioral: digital phenotyping, cognitive tests
- Physiological: heart rate, HRV, cortisol

Fusion strategies:
1. Early fusion: Concatenate features before model
2. Late fusion: Train separate models, ensemble predictions
3. Intermediate fusion: Multi-branch networks with cross-modal attention
4. Hierarchical fusion: Progressive integration

Applications:
- Depression diagnosis (EEG + digital phenotype + clinical)
- Schizophrenia prediction (genetics + fMRI + symptoms)
- Treatment response prediction (multimodal baseline → outcome)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class Modality(str, Enum):
    """Available data modalities"""
    EEG = "eeg"
    FMRI = "fmri"
    MRI_STRUCTURAL = "mri_structural"
    GENETICS = "genetics"
    CLINICAL = "clinical"
    DIGITAL_PHENOTYPE = "digital_phenotype"
    COGNITIVE = "cognitive"
    PHYSIOLOGICAL = "physiological"


class FusionStrategy(str, Enum):
    """Fusion strategies"""
    EARLY = "early"  # Concatenate features
    LATE = "late"  # Ensemble predictions
    INTERMEDIATE = "intermediate"  # Cross-modal attention
    HIERARCHICAL = "hierarchical"  # Progressive integration


@dataclass
class ModalityData:
    """Container for single modality data"""
    modality: Modality
    features: np.ndarray  # (n_samples, n_features)
    feature_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0  # Data quality (0-1)


@dataclass
class MultimodalSample:
    """Complete multimodal sample for one subject"""
    subject_id: str
    modalities: Dict[Modality, ModalityData]
    label: Optional[int] = None  # For supervised learning
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModalityEncoder(nn.Module):
    """
    Modality-specific encoder.
    
    Each modality has its own encoder to learn optimal representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism.
    
    Allows different modalities to attend to each other,
    learning complementary information.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, hidden_dim) - target modality
            key: (batch, hidden_dim) - source modality
            value: (batch, hidden_dim) - source modality
        
        Returns:
            attended: (batch, hidden_dim)
        """
        # Reshape for multihead attention
        query = query.unsqueeze(1)  # (batch, 1, hidden_dim)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        
        # Attention
        attended, _ = self.multihead_attn(query, key, value)
        attended = attended.squeeze(1)  # (batch, hidden_dim)
        
        # Residual + norm
        output = self.norm(query.squeeze(1) + attended)
        
        return output


class EarlyFusionModel(nn.Module):
    """
    Early fusion: Concatenate all modality features before model.
    
    Simple but effective baseline.
    """
    
    def __init__(
        self,
        input_dims: Dict[Modality, int],
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        total_input_dim = sum(input_dims.values())
        
        self.fusion_network = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, modality_features: Dict[Modality, torch.Tensor]) -> torch.Tensor:
        # Concatenate all modalities
        features = torch.cat(list(modality_features.values()), dim=1)
        logits = self.fusion_network(features)
        return logits


class IntermediateFusionModel(nn.Module):
    """
    Intermediate fusion: Modality-specific encoders + cross-modal attention.
    
    Allows learning modality-specific and cross-modal patterns.
    """
    
    def __init__(
        self,
        input_dims: Dict[Modality, int],
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.modalities = list(input_dims.keys())
        
        # Modality-specific encoders
        self.encoders = nn.ModuleDict({
            modality.value: ModalityEncoder(
                input_dim=input_dims[modality],
                hidden_dim=hidden_dim * 2,
                output_dim=hidden_dim,
                dropout=dropout
            )
            for modality in self.modalities
        })
        
        # Cross-modal attention (each modality attends to others)
        self.cross_attention = nn.ModuleDict({
            modality.value: CrossModalAttention(hidden_dim=hidden_dim)
            for modality in self.modalities
        })
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * len(self.modalities), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, modality_features: Dict[Modality, torch.Tensor]) -> torch.Tensor:
        # 1. Encode each modality
        encoded = {}
        for modality in self.modalities:
            encoded[modality] = self.encoders[modality.value](
                modality_features[modality]
            )
        
        # 2. Cross-modal attention
        attended = {}
        for target_modality in self.modalities:
            # Target modality attends to all other modalities
            other_modalities = [m for m in self.modalities if m != target_modality]
            
            if len(other_modalities) > 0:
                # Average other modalities as key/value
                other_encoded = torch.stack([
                    encoded[m] for m in other_modalities
                ], dim=1).mean(dim=1)
                
                attended[target_modality] = self.cross_attention[target_modality.value](
                    query=encoded[target_modality],
                    key=other_encoded,
                    value=other_encoded
                )
            else:
                attended[target_modality] = encoded[target_modality]
        
        # 3. Concatenate attended representations
        fused = torch.cat(list(attended.values()), dim=1)
        
        # 4. Classification
        logits = self.classifier(fused)
        
        return logits


class HierarchicalFusionModel(nn.Module):
    """
    Hierarchical fusion: Progressive integration by modality groups.
    
    Groups:
    1. Neuroimaging (EEG + fMRI + MRI)
    2. Biological (genetics + physiological)
    3. Behavioral (digital phenotype + cognitive)
    4. Clinical
    
    Then fuse groups.
    """
    
    def __init__(
        self,
        input_dims: Dict[Modality, int],
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Define modality groups
        self.groups = {
            'neuroimaging': [Modality.EEG, Modality.FMRI, Modality.MRI_STRUCTURAL],
            'biological': [Modality.GENETICS, Modality.PHYSIOLOGICAL],
            'behavioral': [Modality.DIGITAL_PHENOTYPE, Modality.COGNITIVE],
            'clinical': [Modality.CLINICAL]
        }
        
        # Group encoders
        self.group_encoders = nn.ModuleDict()
        for group_name, modalities in self.groups.items():
            available_modalities = [m for m in modalities if m in input_dims]
            if available_modalities:
                group_input_dim = sum(input_dims[m] for m in available_modalities)
                self.group_encoders[group_name] = nn.Sequential(
                    nn.Linear(group_input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
        
        # Final fusion
        num_groups = len(self.group_encoders)
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_groups, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, modality_features: Dict[Modality, torch.Tensor]) -> torch.Tensor:
        # 1. Encode each group
        group_representations = []
        
        for group_name, modalities in self.groups.items():
            if group_name in self.group_encoders:
                # Concatenate modalities in group
                group_features = torch.cat([
                    modality_features[m]
                    for m in modalities
                    if m in modality_features
                ], dim=1)
                
                # Encode group
                group_repr = self.group_encoders[group_name](group_features)
                group_representations.append(group_repr)
        
        # 2. Fuse groups
        fused = torch.cat(group_representations, dim=1)
        logits = self.final_fusion(fused)
        
        return logits


class MultimodalFusion:
    """
    Multimodal fusion system for computational psychiatry.
    
    Supports multiple fusion strategies and handles missing modalities.
    """
    
    def __init__(
        self,
        fusion_strategy: FusionStrategy = FusionStrategy.INTERMEDIATE,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        if not all([HAS_TORCH, HAS_SKLEARN]):
            raise ImportError("PyTorch and scikit-learn required")
        
        self.fusion_strategy = fusion_strategy
        self.device = device
        self.model: Optional[nn.Module] = None
        self.scalers: Dict[Modality, StandardScaler] = {}
        
        logger.info(f"Multimodal Fusion initialized: {fusion_strategy}")
    
    def preprocess_modality(
        self,
        data: ModalityData,
        fit: bool = False
    ) -> np.ndarray:
        """
        Preprocess single modality data.
        
        Args:
            data: Modality data
            fit: Whether to fit scaler (training) or use existing (inference)
        
        Returns:
            Preprocessed features
        """
        modality = data.modality
        features = data.features
        
        # Standardization
        if fit:
            self.scalers[modality] = StandardScaler()
            features = self.scalers[modality].fit_transform(features)
        else:
            if modality in self.scalers:
                features = self.scalers[modality].transform(features)
        
        return features
    
    def build_model(
        self,
        input_dims: Dict[Modality, int],
        num_classes: int = 2,
        hidden_dim: int = 128
    ):
        """
        Build fusion model based on strategy.
        
        Args:
            input_dims: Input dimensions per modality
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension
        """
        logger.info(f"Building {self.fusion_strategy} fusion model...")
        
        if self.fusion_strategy == FusionStrategy.EARLY:
            self.model = EarlyFusionModel(
                input_dims=input_dims,
                hidden_dim=hidden_dim * 4,
                num_classes=num_classes
            )
        
        elif self.fusion_strategy == FusionStrategy.INTERMEDIATE:
            self.model = IntermediateFusionModel(
                input_dims=input_dims,
                hidden_dim=hidden_dim,
                num_classes=num_classes
            )
        
        elif self.fusion_strategy == FusionStrategy.HIERARCHICAL:
            self.model = HierarchicalFusionModel(
                input_dims=input_dims,
                hidden_dim=hidden_dim,
                num_classes=num_classes
            )
        
        else:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")
        
        self.model = self.model.to(self.device)
        logger.info(f"Model built with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train(
        self,
        train_samples: List[MultimodalSample],
        val_samples: Optional[List[MultimodalSample]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ) -> Dict[str, List[float]]:
        """
        Train multimodal fusion model.
        
        Args:
            train_samples: Training samples
            val_samples: Validation samples
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        
        Returns:
            Training history
        """
        logger.info(f"Training for {epochs} epochs...")
        
        # Prepare data
        # (Simplified - full implementation would use DataLoader)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_loss': [], 'train_acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Training loop (simplified)
            for sample in train_samples:
                # Extract features
                modality_features = {}
                for modality, data in sample.modalities.items():
                    features = self.preprocess_modality(data, fit=(epoch == 0))
                    modality_features[modality] = torch.from_numpy(features).float().to(self.device)
                
                label = torch.tensor([sample.label]).long().to(self.device)
                
                # Forward
                logits = self.model(modality_features)
                loss = criterion(logits, label)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Metrics
                epoch_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += 1
            
            # Log
            avg_loss = epoch_loss / len(train_samples)
            accuracy = correct / total
            
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(accuracy)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        return history
    
    def predict(self, sample: MultimodalSample) -> Dict[str, Any]:
        """
        Predict on multimodal sample.
        
        Args:
            sample: Multimodal sample
        
        Returns:
            Prediction with probabilities and interpretation
        """
        self.model.eval()
        
        with torch.no_grad():
            # Extract features
            modality_features = {}
            for modality, data in sample.modalities.items():
                features = self.preprocess_modality(data, fit=False)
                modality_features[modality] = torch.from_numpy(features).float().to(self.device)
            
            # Predict
            logits = self.model(modality_features)
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        return {
            'prediction': pred_class,
            'confidence': confidence,
            'probabilities': probs[0].cpu().numpy().tolist(),
            'modalities_used': list(modality_features.keys())
        }
    
    def get_modality_importance(
        self,
        sample: MultimodalSample
    ) -> Dict[Modality, float]:
        """
        Estimate modality importance via ablation.
        
        Args:
            sample: Multimodal sample
        
        Returns:
            Importance scores per modality
        """
        # Baseline prediction (all modalities)
        baseline_pred = self.predict(sample)
        baseline_conf = baseline_pred['confidence']
        
        importance = {}
        
        # Ablate each modality
        for modality in sample.modalities.keys():
            # Create ablated sample
            ablated_sample = MultimodalSample(
                subject_id=sample.subject_id,
                modalities={
                    m: data for m, data in sample.modalities.items()
                    if m != modality
                },
                label=sample.label
            )
            
            # Predict without this modality
            ablated_pred = self.predict(ablated_sample)
            ablated_conf = ablated_pred['confidence']
            
            # Importance = drop in confidence
            importance[modality] = baseline_conf - ablated_conf
        
        return importance


# Example usage
if __name__ == "__main__":
    # Example: Depression prediction from EEG + digital phenotype + clinical
    
    # Create dummy data
    sample = MultimodalSample(
        subject_id="patient_001",
        modalities={
            Modality.EEG: ModalityData(
                modality=Modality.EEG,
                features=np.random.randn(1, 100)  # 100 EEG features
            ),
            Modality.DIGITAL_PHENOTYPE: ModalityData(
                modality=Modality.DIGITAL_PHENOTYPE,
                features=np.random.randn(1, 50)  # 50 digital phenotype features
            ),
            Modality.CLINICAL: ModalityData(
                modality=Modality.CLINICAL,
                features=np.random.randn(1, 20)  # 20 clinical features
            )
        },
        label=1  # 1 = depression
    )
    
    # Initialize fusion
    fusion = MultimodalFusion(fusion_strategy=FusionStrategy.INTERMEDIATE)
    
    # Build model
    input_dims = {
        Modality.EEG: 100,
        Modality.DIGITAL_PHENOTYPE: 50,
        Modality.CLINICAL: 20
    }
    fusion.build_model(input_dims, num_classes=2)
    
    # Predict
    result = fusion.predict(sample)
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    # Modality importance
    importance = fusion.get_modality_importance(sample)
    print("\n--- Modality Importance ---")
    for mod, score in importance.items():
        print(f"{mod}: {score:.3f}")

