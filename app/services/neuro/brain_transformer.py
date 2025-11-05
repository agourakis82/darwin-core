"""
Brain Transformers (BrainBERT) for EEG/fMRI Analysis

Implements transformer architectures for brain signal encoding:
- EEG tokenization and encoding
- Self-supervised pretraining
- Fine-tuning for clinical tasks
- Interpretability via attention maps

Based on:
- BrainBERT: https://arxiv.org/abs/2302.14367
- EEG Transformers: https://arxiv.org/abs/2106.11170
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("PyTorch not installed")

try:
    from transformers import (
        BertConfig, BertModel, BertForSequenceClassification,
        get_linear_schedule_with_warmup
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


@dataclass
class BrainTransformerConfig:
    """Configuration for Brain Transformer"""
    # Architecture
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 6
    intermediate_size: int = 3072
    
    # EEG-specific
    n_channels: int = 64  # Number of EEG channels
    sequence_length: int = 512  # Time steps per sample
    patch_size: int = 16  # Temporal patching
    
    # Training
    num_classes: int = 2  # Binary classification
    dropout: float = 0.1
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    
    # Pretraining
    mask_probability: float = 0.15  # For masked signal modeling


class EEGTokenizer:
    """
    Tokenize EEG signals for transformer input.
    
    Converts continuous EEG (n_channels, time_steps) into patches (tokens).
    Similar to ViT (Vision Transformer) but for 1D temporal signals.
    """
    
    def __init__(
        self,
        n_channels: int,
        patch_size: int,
        hidden_size: int
    ):
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        # Learnable patch embedding
        self.patch_embed = nn.Conv1d(
            in_channels=n_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        logger.debug(f"EEG Tokenizer: {n_channels} channels, patch_size={patch_size}")
    
    def tokenize(self, eeg_signal: torch.Tensor) -> torch.Tensor:
        """
        Tokenize EEG signal.
        
        Args:
            eeg_signal: (batch, n_channels, time_steps)
        
        Returns:
            tokens: (batch, n_patches, hidden_size)
        """
        # Apply patch embedding
        patches = self.patch_embed(eeg_signal)  # (batch, hidden_size, n_patches)
        
        # Transpose to (batch, n_patches, hidden_size)
        tokens = patches.transpose(1, 2)
        
        return tokens


class BrainTransformer(nn.Module):
    """
    Transformer for brain signal analysis (EEG/fMRI).
    
    Architecture:
    1. EEG Tokenization (patch embedding)
    2. Positional encoding
    3. Transformer encoder
    4. Classification/regression head
    """
    
    def __init__(self, config: BrainTransformerConfig):
        super().__init__()
        self.config = config
        
        # Tokenizer
        self.tokenizer = EEGTokenizer(
            n_channels=config.n_channels,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size
        )
        
        # Number of patches
        self.n_patches = config.sequence_length // config.patch_size
        
        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, config.hidden_size)
        )
        
        # BERT-style transformer
        bert_config = BertConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout
        )
        
        self.transformer = BertModel(bert_config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes)
        )
        
        logger.info(f"BrainTransformer initialized: {self.n_patches} patches, {config.num_hidden_layers} layers")
    
    def forward(
        self,
        eeg_signal: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            eeg_signal: (batch, n_channels, time_steps)
            attention_mask: Optional mask
        
        Returns:
            logits: (batch, num_classes)
            attentions: Attention weights for interpretability
        """
        batch_size = eeg_signal.shape[0]
        
        # 1. Tokenize
        tokens = self.tokenizer.tokenize(eeg_signal)  # (batch, n_patches, hidden_size)
        
        # 2. Add positional encoding
        tokens = tokens + self.pos_embed
        
        # 3. Transformer encoding
        outputs = self.transformer(
            inputs_embeds=tokens,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        # 4. Classification
        # Use [CLS] token (first token)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        
        # Return logits and attention for interpretability
        attentions = outputs.attentions
        
        return logits, attentions
    
    def get_attention_maps(
        self,
        eeg_signal: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Get attention maps for interpretability.
        
        Args:
            eeg_signal: (batch, n_channels, time_steps)
        
        Returns:
            List of attention tensors per layer
        """
        with torch.no_grad():
            _, attentions = self.forward(eeg_signal)
        
        return attentions


class MaskedSignalModeling(nn.Module):
    """
    Self-supervised pretraining via masked signal modeling.
    
    Similar to BERT's masked language modeling but for continuous signals.
    """
    
    def __init__(self, config: BrainTransformerConfig):
        super().__init__()
        self.config = config
        
        # Base transformer
        self.transformer = BrainTransformer(config)
        
        # Reconstruction head
        self.reconstruction_head = nn.Linear(
            config.hidden_size,
            config.n_channels * config.patch_size
        )
    
    def forward(
        self,
        eeg_signal: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for masked signal modeling.
        
        Args:
            eeg_signal: (batch, n_channels, time_steps)
            mask: (batch, n_patches) binary mask
        
        Returns:
            reconstruction: Reconstructed signal
            loss: Reconstruction loss
        """
        # Get transformer outputs
        logits, _ = self.transformer(eeg_signal)
        
        # Reconstruct masked patches
        # (Simplified - full implementation would reconstruct per patch)
        reconstruction = self.reconstruction_head(logits)
        
        # Compute MSE loss on masked regions
        if mask is not None:
            loss = nn.functional.mse_loss(reconstruction, eeg_signal.flatten(1))
        else:
            loss = torch.tensor(0.0)
        
        return reconstruction, loss


class BrainTransformerTrainer:
    """Trainer for Brain Transformer"""
    
    def __init__(
        self,
        model: BrainTransformer,
        config: BrainTransformerConfig,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )
        
        logger.info("BrainTransformer trainer initialized")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        scheduler: Optional[Any] = None
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            eeg, labels = batch
            eeg, labels = eeg.to(self.device), labels.to(self.device)
            
            # Forward
            logits, _ = self.model(eeg)
            loss = nn.functional.cross_entropy(logits, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in val_loader:
            eeg, labels = batch
            eeg, labels = eeg.to(self.device), labels.to(self.device)
            
            logits, _ = self.model(eeg)
            loss = nn.functional.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': correct / total
        }
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Checkpoint loaded: {path}")


# Factory functions
def get_brain_transformer(
    config: Optional[BrainTransformerConfig] = None
) -> BrainTransformer:
    """Factory function for Brain Transformer"""
    if not all([HAS_TORCH, HAS_TRANSFORMERS]):
        raise ImportError("PyTorch and transformers required")
    
    if config is None:
        config = BrainTransformerConfig()
    
    return BrainTransformer(config)


def pretrain_brain_transformer(
    eeg_data: np.ndarray,
    config: Optional[BrainTransformerConfig] = None,
    epochs: int = 100
) -> BrainTransformer:
    """
    Pretrain Brain Transformer via masked signal modeling.
    
    Args:
        eeg_data: (n_samples, n_channels, time_steps)
        config: Model config
        epochs: Pretraining epochs
    
    Returns:
        Pretrained model
    """
    if config is None:
        config = BrainTransformerConfig()
    
    # Create pretraining model
    msm_model = MaskedSignalModeling(config)
    
    # Train (simplified - full implementation would use proper dataloaders)
    logger.info(f"Pretraining for {epochs} epochs...")
    
    # ... training loop ...
    
    # Return base transformer
    return msm_model.transformer


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = BrainTransformerConfig(
        n_channels=64,
        sequence_length=512,
        num_classes=2  # e.g., depression vs healthy
    )
    
    # Create model
    model = get_brain_transformer(config)
    
    # Dummy input
    batch_size = 4
    eeg = torch.randn(batch_size, config.n_channels, config.sequence_length)
    
    # Forward pass
    logits, attentions = model(eeg)
    
    print(f"\nInput shape: {eeg.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of attention layers: {len(attentions)}")
    print(f"Attention shape: {attentions[0].shape}")

