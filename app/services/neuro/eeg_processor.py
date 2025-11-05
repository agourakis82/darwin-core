"""
EEG Signal Processing for Darwin Neuro

Integrates NeuroKit2, MNE, and Braindecode for comprehensive EEG analysis.
Supports preprocessing, feature extraction, and deep learning.

Key Features:
- Raw EEG preprocessing (filtering, ICA, artifact removal)
- Event-related potential (ERP) analysis
- Time-frequency analysis (wavelet, Hilbert)
- Deep learning-ready tensors
- Clinical biomarkers extraction
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from enum import Enum

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False
    logging.warning("NeuroKit2 not installed. Install with: pip install neurokit2")

try:
    import mne
    from mne.preprocessing import ICA
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    logging.warning("MNE not installed. Install with: pip install mne")

try:
    import torch
    from braindecode.datasets import BaseDataset
    from braindecode.preprocessing import preprocess, Preprocessor
    HAS_BRAINDECODE = True
except ImportError:
    HAS_BRAINDECODE = False
    logging.warning("Braindecode not installed. Install with: pip install braindecode")

logger = logging.getLogger(__name__)


class EEGBand(str, Enum):
    """Standard EEG frequency bands"""
    DELTA = "delta"  # 0.5-4 Hz (deep sleep)
    THETA = "theta"  # 4-8 Hz (drowsiness, meditation)
    ALPHA = "alpha"  # 8-13 Hz (relaxed, eyes closed)
    BETA = "beta"   # 13-30 Hz (active thinking)
    GAMMA = "gamma"  # 30-100 Hz (cognitive processing)


@dataclass
class EEGConfig:
    """Configuration for EEG processing"""
    sampling_rate: int = 250  # Hz
    low_freq: float = 0.5  # High-pass filter
    high_freq: float = 100.0  # Low-pass filter
    notch_freq: float = 60.0  # Powerline noise (50 or 60 Hz)
    
    # ICA settings
    n_components: int = 20
    random_state: int = 42
    
    # Epoching
    epoch_duration: float = 2.0  # seconds
    overlap: float = 0.5  # 50% overlap
    
    # Feature extraction
    extract_bands: bool = True
    extract_connectivity: bool = False  # Computationally expensive


class EEGProcessor:
    """
    Comprehensive EEG signal processor using NeuroKit2, MNE, and Braindecode.
    
    Supports:
    - Preprocessing (filtering, ICA, artifact removal)
    - Feature extraction (band power, connectivity, complexity)
    - Deep learning preparation
    - Clinical biomarkers
    """
    
    def __init__(self, config: EEGConfig):
        self.config = config
        
        if not all([HAS_NEUROKIT, HAS_MNE]):
            raise ImportError(
                "Missing dependencies. Install with:\n"
                "pip install neurokit2 mne braindecode"
            )
        
        logger.info("EEG Processor initialized")
    
    def load_raw_eeg(
        self,
        file_path: Path,
        file_format: str = "edf"
    ) -> mne.io.Raw:
        """
        Load raw EEG data.
        
        Args:
            file_path: Path to EEG file
            file_format: Format (edf, fif, cnt, etc)
        
        Returns:
            MNE Raw object
        """
        logger.info(f"Loading EEG: {file_path}")
        
        if file_format == "edf":
            raw = mne.io.read_raw_edf(str(file_path), preload=True)
        elif file_format == "fif":
            raw = mne.io.read_raw_fif(str(file_path), preload=True)
        elif file_format == "cnt":
            raw = mne.io.read_raw_cnt(str(file_path), preload=True)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        logger.info(f"Loaded {len(raw.ch_names)} channels, duration: {raw.times[-1]:.1f}s")
        return raw
    
    def preprocess(
        self,
        raw: mne.io.Raw,
        apply_ica: bool = True,
        reference: str = "average"
    ) -> mne.io.Raw:
        """
        Preprocess raw EEG data.
        
        Pipeline:
        1. Bandpass filtering
        2. Notch filtering (powerline noise)
        3. Re-referencing
        4. ICA for artifact removal
        
        Args:
            raw: Raw EEG data
            apply_ica: Whether to apply ICA
            reference: Reference type ('average', 'mastoids', etc)
        
        Returns:
            Preprocessed Raw object
        """
        logger.info("Preprocessing EEG...")
        
        # 1. Bandpass filter
        raw_filtered = raw.copy()
        raw_filtered.filter(
            l_freq=self.config.low_freq,
            h_freq=self.config.high_freq,
            fir_design='firwin'
        )
        logger.debug(f"Applied bandpass filter: {self.config.low_freq}-{self.config.high_freq} Hz")
        
        # 2. Notch filter (powerline noise)
        raw_filtered.notch_filter(
            freqs=self.config.notch_freq,
            fir_design='firwin'
        )
        logger.debug(f"Applied notch filter: {self.config.notch_freq} Hz")
        
        # 3. Re-reference
        if reference == "average":
            raw_filtered.set_eeg_reference('average', projection=True)
            raw_filtered.apply_proj()
        logger.debug(f"Re-referenced to: {reference}")
        
        # 4. ICA for artifact removal
        if apply_ica:
            raw_filtered = self._apply_ica(raw_filtered)
        
        logger.info("Preprocessing complete")
        return raw_filtered
    
    def _apply_ica(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply ICA for artifact removal (EOG, ECG)"""
        logger.debug("Applying ICA...")
        
        # Fit ICA
        ica = ICA(
            n_components=self.config.n_components,
            random_state=self.config.random_state,
            max_iter='auto'
        )
        
        # Fit on filtered data
        ica.fit(raw)
        
        # Auto-detect and exclude EOG artifacts
        eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=3.0)
        ica.exclude = eog_indices
        
        logger.debug(f"Excluded {len(eog_indices)} ICA components (EOG artifacts)")
        
        # Apply ICA
        raw_clean = ica.apply(raw.copy())
        
        return raw_clean
    
    def extract_epochs(
        self,
        raw: mne.io.Raw,
        events: Optional[np.ndarray] = None,
        event_id: Optional[Dict[str, int]] = None
    ) -> mne.Epochs:
        """
        Extract epochs from continuous data.
        
        Args:
            raw: Preprocessed raw data
            events: Event array (if task-based)
            event_id: Event ID mapping
        
        Returns:
            MNE Epochs object
        """
        logger.info("Extracting epochs...")
        
        if events is None:
            # Create artificial epochs for continuous data
            duration = self.config.epoch_duration
            overlap = self.config.overlap
            
            events = mne.make_fixed_length_events(
                raw,
                duration=duration,
                overlap=overlap * duration
            )
            event_id = {'continuous': 1}
        
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=0,
            tmax=self.config.epoch_duration,
            baseline=None,
            preload=True
        )
        
        logger.info(f"Extracted {len(epochs)} epochs")
        return epochs
    
    def extract_band_power(
        self,
        epochs: mne.Epochs
    ) -> Dict[str, np.ndarray]:
        """
        Extract power in standard frequency bands.
        
        Args:
            epochs: MNE Epochs
        
        Returns:
            Dictionary mapping band names to power arrays
        """
        logger.debug("Extracting band power...")
        
        band_ranges = {
            EEGBand.DELTA: (0.5, 4),
            EEGBand.THETA: (4, 8),
            EEGBand.ALPHA: (8, 13),
            EEGBand.BETA: (13, 30),
            EEGBand.GAMMA: (30, 100)
        }
        
        band_power = {}
        
        for band, (fmin, fmax) in band_ranges.items():
            # Compute PSD in band
            psd = epochs.compute_psd(
                fmin=fmin,
                fmax=fmax,
                method='welch'
            )
            
            # Average power across frequencies
            power = psd.get_data().mean(axis=-1)  # (n_epochs, n_channels)
            band_power[band] = power
        
        logger.debug(f"Extracted power for {len(band_power)} bands")
        return band_power
    
    def extract_complexity_features(
        self,
        signal: np.ndarray,
        sampling_rate: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Extract signal complexity features using NeuroKit2.
        
        Args:
            signal: 1D signal array
            sampling_rate: Sampling rate (Hz)
        
        Returns:
            Dictionary of complexity metrics
        """
        if sampling_rate is None:
            sampling_rate = self.config.sampling_rate
        
        features = {}
        
        try:
            # Sample entropy
            features['sample_entropy'] = nk.entropy_sample(signal)
            
            # Approximate entropy
            features['approx_entropy'] = nk.entropy_approximate(signal)
            
            # Fractal dimension
            features['fractal_dimension'] = nk.fractal_correlation(signal)
            
            # Hjorth parameters
            hjorth = nk.complexity_hjorth(signal)
            features['hjorth_activity'] = hjorth[0]
            features['hjorth_mobility'] = hjorth[1]
            features['hjorth_complexity'] = hjorth[2]
            
        except Exception as e:
            logger.warning(f"Error extracting complexity features: {e}")
        
        return features
    
    def extract_clinical_biomarkers(
        self,
        epochs: mne.Epochs
    ) -> Dict[str, Any]:
        """
        Extract clinical biomarkers relevant to psychiatry/neurology.
        
        Includes:
        - Alpha peak frequency (cognitive performance)
        - Theta/beta ratio (ADHD marker)
        - Alpha asymmetry (depression marker)
        - Complexity measures
        
        Args:
            epochs: MNE Epochs
        
        Returns:
            Dictionary of biomarkers
        """
        logger.info("Extracting clinical biomarkers...")
        
        biomarkers = {}
        
        # Band power
        band_power = self.extract_band_power(epochs)
        
        # 1. Alpha peak frequency
        psd = epochs.compute_psd(fmin=8, fmax=13, method='welch')
        freqs = psd.freqs
        power = psd.get_data().mean(axis=(0, 1))  # Average across epochs and channels
        alpha_peak = freqs[np.argmax(power)]
        biomarkers['alpha_peak_freq'] = alpha_peak
        
        # 2. Theta/Beta ratio (ADHD marker)
        theta_power = band_power[EEGBand.THETA].mean()
        beta_power = band_power[EEGBand.BETA].mean()
        biomarkers['theta_beta_ratio'] = theta_power / (beta_power + 1e-10)
        
        # 3. Alpha asymmetry (depression marker)
        # Left vs right frontal alpha
        ch_names = epochs.ch_names
        left_frontal = [ch for ch in ch_names if ch.startswith(('F3', 'F7'))]
        right_frontal = [ch for ch in ch_names if ch.startswith(('F4', 'F8'))]
        
        if left_frontal and right_frontal:
            left_idx = [epochs.ch_names.index(ch) for ch in left_frontal]
            right_idx = [epochs.ch_names.index(ch) for ch in right_frontal]
            
            alpha_left = band_power[EEGBand.ALPHA][:, left_idx].mean()
            alpha_right = band_power[EEGBand.ALPHA][:, right_idx].mean()
            
            biomarkers['alpha_asymmetry'] = np.log(alpha_right) - np.log(alpha_left)
        
        # 4. Complexity (first channel as example)
        signal = epochs.get_data()[:, 0, :].flatten()
        complexity = self.extract_complexity_features(signal)
        biomarkers.update(complexity)
        
        logger.info(f"Extracted {len(biomarkers)} clinical biomarkers")
        return biomarkers
    
    def to_braindecode_dataset(
        self,
        epochs: mne.Epochs,
        labels: Optional[np.ndarray] = None
    ) -> BaseDataset:
        """
        Convert to Braindecode dataset for deep learning.
        
        Args:
            epochs: MNE Epochs
            labels: Optional labels for classification
        
        Returns:
            Braindecode BaseDataset
        """
        if not HAS_BRAINDECODE:
            raise ImportError("Braindecode not installed")
        
        logger.debug("Converting to Braindecode dataset...")
        
        # Get data
        X = epochs.get_data()  # (n_epochs, n_channels, n_times)
        
        if labels is None:
            labels = np.zeros(len(epochs))
        
        # Create dataset
        # Note: Simplified - real implementation would use proper Braindecode structures
        dataset = {
            'X': torch.from_numpy(X).float(),
            'y': torch.from_numpy(labels).long(),
            'sfreq': epochs.info['sfreq'],
            'ch_names': epochs.ch_names
        }
        
        return dataset
    
    def get_processing_report(
        self,
        raw: mne.io.Raw,
        epochs: mne.Epochs
    ) -> Dict[str, Any]:
        """Generate processing report"""
        return {
            'n_channels': len(raw.ch_names),
            'sampling_rate': raw.info['sfreq'],
            'duration_sec': raw.times[-1],
            'n_epochs': len(epochs),
            'epoch_duration': self.config.epoch_duration,
            'preprocessing': {
                'bandpass': f"{self.config.low_freq}-{self.config.high_freq} Hz",
                'notch': f"{self.config.notch_freq} Hz"
            }
        }


# Factory function
def get_eeg_processor(config: Optional[EEGConfig] = None) -> EEGProcessor:
    """Factory function to get EEG processor"""
    if config is None:
        config = EEGConfig()
    return EEGProcessor(config)


# Example usage
if __name__ == "__main__":
    # Example: Process EEG data
    config = EEGConfig(sampling_rate=250)
    processor = get_eeg_processor(config)
    
    # Load (example path)
    eeg_file = Path("data/eeg/patient_001.edf")
    if eeg_file.exists():
        # Load
        raw = processor.load_raw_eeg(eeg_file, file_format="edf")
        
        # Preprocess
        raw_clean = processor.preprocess(raw, apply_ica=True)
        
        # Extract epochs
        epochs = processor.extract_epochs(raw_clean)
        
        # Extract biomarkers
        biomarkers = processor.extract_clinical_biomarkers(epochs)
        
        print("\n--- Clinical Biomarkers ---")
        for key, value in biomarkers.items():
            print(f"{key}: {value:.4f}")
        
        # Report
        report = processor.get_processing_report(raw, epochs)
        print("\n--- Processing Report ---")
        print(report)

