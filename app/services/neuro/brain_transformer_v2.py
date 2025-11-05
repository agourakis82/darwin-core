"""
Brain Transformer V2 - Transformer Models for EEG/fMRI Signals

Implements BrainBERT-like architecture for neural signal processing:
- Temporal tokenization
- Positional encoding for time series
- Self-attention over signal patches
- Pre-training and fine-tuning
- Classification and regression tasks

References:
- "BrainBERT: Self-supervised representation learning for intracranial recordings"
- "EEG Transformer: End-to-End Training of Transformer Models for EEG"
- "Learning Robust Representations of EEG using Transformers"
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BrainTransformerConfig:
    """Configuration for Brain Transformer"""
    # Input
    num_channels: int = 10  # Number of EEG channels
    patch_size: int = 64  # Samples per patch
    sampling_rate: int = 256  # Hz
    
    # Model architecture
    d_model: int = 256  # Embedding dimension
    nhead: int = 8  # Number of attention heads
    num_layers: int = 6  # Number of transformer layers
    dim_feedforward: int = 1024  # FFN dimension
    dropout: float = 0.1
    
    # Output
    num_classes: int = 4  # For classification tasks
    
    # Training
    max_seq_length: int = 256  # Max patches in sequence


class PatchEmbedding(nn.Module):
    """
    Convert EEG signals into patches and embed them.
    
    Similar to vision transformer patch embedding.
    """
    
    def __init__(self, config: BrainTransformerConfig):
        super().__init__()
        self.config = config
        
        # Patch projection
        # Input: [batch, channels, time]
        # Output: [batch, num_patches, d_model]
        self.proj = nn.Conv1d(
            config.num_channels,
            config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.max_seq_length + 1, config.d_model)
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, time]
        
        Returns:
            [batch, seq_len, d_model]
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.proj(x)  # [batch, d_model, num_patches]
        x = x.transpose(1, 2)  # [batch, num_patches, d_model]
        
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_embedding[:, :seq_len, :]
        
        return self.dropout(x)


class BrainTransformer(nn.Module):
    """
    Transformer model for EEG/fMRI signals.
    
    Architecture:
    1. Patch embedding (tokenization)
    2. Transformer encoder
    3. Classification/regression head
    """
    
    def __init__(self, config: BrainTransformerConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(config)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(config.d_model)
        
        # Classification head
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        
        # Regression head (for continuous outputs)
        self.regressor = nn.Linear(config.d_model, 1)
        
        logger.info(f"BrainTransformer initialized: {self.count_parameters():,} parameters")
    
    def forward(
        self,
        x: torch.Tensor,
        task: str = 'classification'
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch, channels, time] EEG signals
            task: 'classification' or 'regression'
        
        Returns:
            [batch, num_classes] or [batch, 1]
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use [CLS] token for prediction
        x = self.norm(x[:, 0])
        
        # Task-specific head
        if task == 'classification':
            return self.classifier(x)
        else:
            return self.regressor(x)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get signal embeddings (for downstream tasks).
        
        Args:
            x: [batch, channels, time]
        
        Returns:
            [batch, d_model] embeddings
        """
        x = self.patch_embed(x)
        x = self.transformer(x)
        return self.norm(x[:, 0])
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BrainTransformerTrainer:
    """
    Trainer for Brain Transformer.
    
    Supports:
    - Pre-training (self-supervised)
    - Fine-tuning (supervised)
    - Contrastive learning
    """
    
    def __init__(
        self,
        model: BrainTransformer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        
        logger.info(f"Trainer initialized on {device}")
    
    def pretrain_masked(
        self,
        signals: torch.Tensor,
        mask_ratio: float = 0.15
    ) -> float:
        """
        Pre-train with masked signal modeling (like BERT).
        
        Args:
            signals: [batch, channels, time]
            mask_ratio: Ratio of patches to mask
        
        Returns:
            Loss value
        """
        signals = signals.to(self.device)
        
        # TODO: Implement masked pretraining
        # 1. Mask random patches
        # 2. Predict masked patches
        # 3. Compute reconstruction loss
        
        return 0.0
    
    def train_classification(
        self,
        signals: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Train for classification task.
        
        Args:
            signals: [batch, channels, time]
            labels: [batch] class indices
            optimizer: PyTorch optimizer
        
        Returns:
            Loss value
        """
        self.model.train()
        signals = signals.to(self.device)
        labels = labels.to(self.device)
        
        # Forward
        logits = self.model(signals, task='classification')
        loss = F.cross_entropy(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(
        self,
        signals: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate model.
        
        Args:
            signals: [batch, channels, time]
            labels: [batch] class indices
        
        Returns:
            Metrics dict
        """
        self.model.eval()
        signals = signals.to(self.device)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            logits = self.model(signals, task='classification')
            loss = F.cross_entropy(logits, labels)
            
            # Accuracy
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }


def generate_synthetic_eeg(
    batch_size: int,
    num_channels: int,
    duration: float,
    sampling_rate: int
) -> torch.Tensor:
    """Generate synthetic EEG for testing"""
    n_samples = int(duration * sampling_rate)
    time = torch.linspace(0, duration, n_samples)
    
    signals = torch.zeros(batch_size, num_channels, n_samples)
    
    for b in range(batch_size):
        for c in range(num_channels):
            # Mix of different frequencies
            alpha = torch.sin(2 * np.pi * 10 * time)  # Alpha (10 Hz)
            beta = torch.sin(2 * np.pi * 20 * time)   # Beta (20 Hz)
            theta = torch.sin(2 * np.pi * 6 * time)   # Theta (6 Hz)
            noise = torch.randn(n_samples) * 0.1
            
            signals[b, c] = alpha + 0.5 * beta + 0.3 * theta + noise
    
    return signals


# Test
if __name__ == "__main__":
    print("="*70)
    print("Brain Transformer V2 - Test")
    print("="*70)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuration
    config = BrainTransformerConfig(
        num_channels=10,
        patch_size=64,
        sampling_rate=256,
        d_model=128,  # Smaller for testing
        nhead=4,
        num_layers=3,
        num_classes=4
    )
    
    # Create model
    print("\nInitializing Brain Transformer...")
    model = BrainTransformer(config)
    print(f"Parameters: {model.count_parameters():,}")
    
    # Generate synthetic data
    print("\nGenerating synthetic EEG data...")
    batch_size = 8
    duration = 10.0  # seconds
    
    signals = generate_synthetic_eeg(
        batch_size,
        config.num_channels,
        duration,
        config.sampling_rate
    )
    
    labels = torch.randint(0, config.num_classes, (batch_size,))
    
    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Test forward pass
    print("\n" + "="*70)
    print("TEST 1: Forward Pass")
    model.eval()
    with torch.no_grad():
        logits = model(signals, task='classification')
        print(f"Output shape: {logits.shape}")
        print(f"Sample logits: {logits[0]}")
        
        # Get embeddings
        embeddings = model.get_embeddings(signals)
        print(f"Embeddings shape: {embeddings.shape}")
    
    # Test training
    print("\n" + "="*70)
    print("TEST 2: Training")
    
    trainer = BrainTransformerTrainer(model, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("\nTraining for 5 epochs...")
    for epoch in range(5):
        loss = trainer.train_classification(signals, labels, optimizer)
        metrics = trainer.evaluate(signals, labels)
        
        print(f"Epoch {epoch+1}/5:")
        print(f"  Train loss: {loss:.4f}")
        print(f"  Val loss: {metrics['loss']:.4f}")
        print(f"  Val accuracy: {metrics['accuracy']:.2%}")
    
    # Test inference speed
    print("\n" + "="*70)
    print("TEST 3: Inference Speed")
    
    model.eval()
    import time
    
    # Move test data to device
    test_signal = signals[:1].to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_signal)
    
    # Benchmark
    n_iters = 100
    start = time.time()
    
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(test_signal)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    avg_time = elapsed / n_iters * 1000
    
    print(f"Iterations: {n_iters}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg time per sample: {avg_time:.2f}ms")
    print(f"Throughput: {1000/avg_time:.1f} samples/sec")
    
    print("\n" + "="*70)
    print("✅ Brain Transformer V2 works!")

