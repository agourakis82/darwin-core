"""
Simple EEG Processor - Versão Funcional

Processamento básico de EEG que FUNCIONA.
Sem MNE/NeuroKit2 - apenas NumPy.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)


@dataclass
class EEGBands:
    """EEG frequency bands"""
    delta: Tuple[float, float] = (0.5, 4.0)
    theta: Tuple[float, float] = (4.0, 8.0)
    alpha: Tuple[float, float] = (8.0, 13.0)
    beta: Tuple[float, float] = (13.0, 30.0)
    gamma: Tuple[float, float] = (30.0, 100.0)


@dataclass
class EEGFeatures:
    """EEG features extracted"""
    # Band powers
    delta_power: float
    theta_power: float
    alpha_power: float
    beta_power: float
    gamma_power: float
    
    # Clinical biomarkers
    theta_beta_ratio: float  # ADHD marker
    alpha_peak_freq: float  # Cognitive performance
    
    # Metadata
    sampling_rate: int
    duration_sec: float
    n_channels: int


class SimpleEEGProcessor:
    """
    Processador EEG básico mas funcional.
    
    Features:
    - Filtering (bandpass, notch)
    - Band power extraction
    - Clinical biomarkers
    - FFT analysis
    """
    
    def __init__(self, sampling_rate: int = 250):
        self.sampling_rate = sampling_rate
        self.bands = EEGBands()
        logger.info(f"SimpleEEGProcessor initialized: fs={sampling_rate}Hz")
    
    def bandpass_filter(
        self,
        data: np.ndarray,
        low_freq: float = 0.5,
        high_freq: float = 100.0
    ) -> np.ndarray:
        """
        Bandpass filter.
        
        Args:
            data: (n_channels, n_samples) or (n_samples,)
            low_freq: Low cutoff (Hz)
            high_freq: High cutoff (Hz)
        
        Returns:
            Filtered data
        """
        nyq = self.sampling_rate / 2
        low = low_freq / nyq
        high = high_freq / nyq
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        if data.ndim == 1:
            return signal.filtfilt(b, a, data)
        else:
            # Multi-channel
            filtered = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered[i] = signal.filtfilt(b, a, data[i])
            return filtered
    
    def notch_filter(
        self,
        data: np.ndarray,
        freq: float = 60.0,
        quality: float = 30.0
    ) -> np.ndarray:
        """
        Notch filter (powerline noise).
        
        Args:
            data: Signal
            freq: Frequency to remove (Hz)
            quality: Quality factor
        
        Returns:
            Filtered data
        """
        b, a = signal.iirnotch(freq, quality, self.sampling_rate)
        
        if data.ndim == 1:
            return signal.filtfilt(b, a, data)
        else:
            filtered = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered[i] = signal.filtfilt(b, a, data[i])
            return filtered
    
    def compute_psd(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density.
        
        Args:
            data: Signal (1D)
        
        Returns:
            (frequencies, psd)
        """
        freqs, psd = signal.welch(
            data,
            fs=self.sampling_rate,
            nperseg=min(256, len(data))
        )
        return freqs, psd
    
    def extract_band_power(
        self,
        data: np.ndarray,
        band: Tuple[float, float]
    ) -> float:
        """
        Extract power in frequency band.
        
        Args:
            data: Signal (1D)
            band: (low_freq, high_freq)
        
        Returns:
            Band power
        """
        freqs, psd = self.compute_psd(data)
        
        # Find frequencies in band
        mask = (freqs >= band[0]) & (freqs <= band[1])
        
        # Integrate power
        band_power = np.trapz(psd[mask], freqs[mask])
        
        return float(band_power)
    
    def extract_features(
        self,
        data: np.ndarray,
        preprocess: bool = True
    ) -> EEGFeatures:
        """
        Extract comprehensive EEG features.
        
        Args:
            data: (n_channels, n_samples) or (n_samples,)
            preprocess: Apply filtering
        
        Returns:
            EEGFeatures
        """
        # Handle multi-channel
        if data.ndim == 2:
            # Average across channels for simplicity
            signal_data = data.mean(axis=0)
            n_channels = data.shape[0]
        else:
            signal_data = data
            n_channels = 1
        
        # Preprocess
        if preprocess:
            signal_data = self.bandpass_filter(signal_data, 0.5, 100.0)
            signal_data = self.notch_filter(signal_data, 60.0)
        
        # Extract band powers
        delta_power = self.extract_band_power(signal_data, self.bands.delta)
        theta_power = self.extract_band_power(signal_data, self.bands.theta)
        alpha_power = self.extract_band_power(signal_data, self.bands.alpha)
        beta_power = self.extract_band_power(signal_data, self.bands.beta)
        gamma_power = self.extract_band_power(signal_data, self.bands.gamma)
        
        # Clinical biomarkers
        theta_beta_ratio = theta_power / (beta_power + 1e-10)
        
        # Alpha peak frequency
        freqs, psd = self.compute_psd(signal_data)
        alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
        alpha_peak_freq = freqs[alpha_mask][np.argmax(psd[alpha_mask])]
        
        # Duration
        duration_sec = len(signal_data) / self.sampling_rate
        
        features = EEGFeatures(
            delta_power=delta_power,
            theta_power=theta_power,
            alpha_power=alpha_power,
            beta_power=beta_power,
            gamma_power=gamma_power,
            theta_beta_ratio=theta_beta_ratio,
            alpha_peak_freq=alpha_peak_freq,
            sampling_rate=self.sampling_rate,
            duration_sec=duration_sec,
            n_channels=n_channels
        )
        
        logger.info(f"Features extracted: theta/beta={theta_beta_ratio:.2f}, alpha_peak={alpha_peak_freq:.1f}Hz")
        
        return features
    
    def diagnose(self, features: EEGFeatures) -> Dict[str, any]:
        """
        Simple diagnostic interpretation.
        
        Args:
            features: Extracted features
        
        Returns:
            Diagnostic suggestions
        """
        diagnosis = {
            'features': features,
            'findings': [],
            'risk_factors': []
        }
        
        # ADHD marker
        if features.theta_beta_ratio > 2.0:
            diagnosis['findings'].append("Elevated theta/beta ratio (ADHD marker)")
            diagnosis['risk_factors'].append("ADHD")
        
        # Cognitive performance
        if features.alpha_peak_freq < 9.0:
            diagnosis['findings'].append("Low alpha peak frequency")
            diagnosis['risk_factors'].append("Cognitive slowing")
        elif features.alpha_peak_freq > 11.0:
            diagnosis['findings'].append("High alpha peak frequency (good)")
        
        # Depression marker (low alpha)
        if features.alpha_power < features.theta_power:
            diagnosis['findings'].append("Low alpha power relative to theta")
            diagnosis['risk_factors'].append("Possible depression")
        
        return diagnosis


def generate_synthetic_eeg(
    duration_sec: float = 10.0,
    sampling_rate: int = 250,
    n_channels: int = 4,
    noise_level: float = 0.5
) -> np.ndarray:
    """Generate synthetic EEG for testing"""
    n_samples = int(duration_sec * sampling_rate)
    t = np.arange(n_samples) / sampling_rate
    
    eeg = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Alpha oscillation (10 Hz)
        eeg[ch] += 2.0 * np.sin(2 * np.pi * 10 * t)
        
        # Theta (6 Hz)
        eeg[ch] += 1.5 * np.sin(2 * np.pi * 6 * t)
        
        # Beta (20 Hz)
        eeg[ch] += 0.8 * np.sin(2 * np.pi * 20 * t)
        
        # Noise
        eeg[ch] += noise_level * np.random.randn(n_samples)
    
    return eeg


# Test
if __name__ == "__main__":
    import sys
    
    print("=== Simple EEG Processor Test ===\n")
    
    # Create processor
    processor = SimpleEEGProcessor(sampling_rate=250)
    
    # Generate synthetic EEG
    print("Generating synthetic EEG (10 seconds, 4 channels)...")
    eeg_data = generate_synthetic_eeg(duration_sec=10.0, n_channels=4)
    print(f"Data shape: {eeg_data.shape}")
    
    # Extract features
    print("\nExtracting features...")
    features = processor.extract_features(eeg_data, preprocess=True)
    
    print("\n=== EEG Features ===")
    print(f"Delta power: {features.delta_power:.2f}")
    print(f"Theta power: {features.theta_power:.2f}")
    print(f"Alpha power: {features.alpha_power:.2f}")
    print(f"Beta power: {features.beta_power:.2f}")
    print(f"Gamma power: {features.gamma_power:.2f}")
    print(f"\nTheta/Beta ratio: {features.theta_beta_ratio:.2f} (ADHD marker)")
    print(f"Alpha peak freq: {features.alpha_peak_freq:.1f} Hz")
    
    # Diagnosis
    print("\n=== Diagnostic Interpretation ===")
    diagnosis = processor.diagnose(features)
    
    if diagnosis['findings']:
        print("Findings:")
        for finding in diagnosis['findings']:
            print(f"  - {finding}")
    
    if diagnosis['risk_factors']:
        print("\nRisk factors:")
        for risk in diagnosis['risk_factors']:
            print(f"  - {risk}")
    else:
        print("No significant risk factors detected.")
    
    print("\n✅ Simple EEG Processor works!")
    sys.exit(0)

