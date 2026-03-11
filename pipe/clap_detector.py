"""
Clap Detector - Multi-feature spectral analysis for clap detection.

This module provides acoustic feature extraction and classification to distinguish
claps (percussive sounds) from speech and other audio events.

Key discriminators:
- Spectral centroid: Claps have higher centroid (>3000 Hz) due to broadband energy
- High-frequency ratio: Claps have significant energy above 4kHz
- Crest factor: Claps show sharp transient peaks (high peak/RMS ratio)
- Duration: Claps are short bursts (10-150ms)
- Attack time: Claps have rapid onset (<8ms rise time)
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.signal import hamming
from scipy.stats import entropy


class ClapDetector:
    """
    Detects claps using spectral and temporal feature analysis.
    
    Unlike simple RMS-based detection, this uses multiple acoustic features
    to distinguish percussive sounds (claps) from speech and other audio events.
    """
    
    # Feature weights for classification scoring
    WEIGHT_CENTROID = 0.25
    WEIGHT_HF_RATIO = 0.25
    WEIGHT_CREST = 0.20
    WEIGHT_DURATION = 0.15
    WEIGHT_ATTACK = 0.15
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize ClapDetector with sample rate.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
        """
        self.sr = sample_rate
        
    @staticmethod
    def compute_spectral_centroid(audio_window: np.ndarray, sr: int) -> float:
        """
        Compute spectral centroid (center of mass of frequency spectrum).
        
        Claps have higher centroid (>3000 Hz) due to broadband energy distribution.
        Speech has lower centroid (<2500 Hz) with formant structure.
        
        Args:
            audio_window: Audio samples as numpy array
            sr: Sample rate in Hz
            
        Returns:
            Spectral centroid in Hz
        """
        # Apply window function to reduce spectral leakage
        window = hamming(len(audio_window))
        windowed = audio_window * window
        
        # Compute FFT
        fft = np.fft.rfft(windowed)
        magnitudes = np.abs(fft)
        frequencies = np.fft.rfftfreq(len(audio_window), 1/sr)
        
        # Avoid division by zero
        total_magnitude = np.sum(magnitudes)
        if total_magnitude < 1e-10:
            return 0.0
            
        centroid = np.sum(frequencies * magnitudes) / total_magnitude
        return centroid
    
    @staticmethod
    def compute_hf_energy_ratio(audio_window: np.ndarray, sr: int, hf_cutoff: float = 4000.0) -> float:
        """
        Compute high-frequency energy ratio (energy above cutoff / total energy).
        
        Claps have significant broadband energy extending to Nyquist frequency.
        Speech has most energy concentrated below 4kHz (human voice range).
        
        Args:
            audio_window: Audio samples as numpy array
            sr: Sample rate in Hz
            hf_cutoff: High-frequency cutoff in Hz (default: 4000)
            
        Returns:
            Ratio of high-frequency energy to total energy (0.0 to 1.0)
        """
        window = hamming(len(audio_window))
        windowed = audio_window * window
        
        fft = np.fft.rfft(windowed)
        magnitudes_squared = np.abs(fft) ** 2
        frequencies = np.fft.rfftfreq(len(audio_window), 1/sr)
        
        hf_energy = np.sum(magnitudes_squared[frequencies > hf_cutoff])
        total_energy = np.sum(magnitudes_squared)
        
        if total_energy < 1e-10:
            return 0.0
            
        return hf_energy / total_energy
    
    @staticmethod
    def compute_spectral_flatness(audio_window: np.ndarray, sr: int) -> float:
        """
        Compute spectral flatness (noise-like vs tonal measure).
        
        Claps have higher flatness (more noise-like, broadband).
        Speech has lower flatness (more tonal with formants).
        
        Args:
            audio_window: Audio samples as numpy array
            sr: Sample rate in Hz
            
        Returns:
            Spectral flatness value (0.0 = purely tonal, 1.0 = white noise)
        """
        window = hamming(len(audio_window))
        windowed = audio_window * window
        
        fft = np.fft.rfft(windowed)
        magnitudes_squared = np.abs(fft) ** 2
        
        # Add small epsilon to avoid log(0)
        magnitudes_squared = magnitudes_squared + 1e-10
        
        # Spectral flatness = geometric mean / arithmetic mean
        geometric_mean = np.exp(np.mean(np.log(magnitudes_squared)))
        arithmetic_mean = np.mean(magnitudes_squared)
        
        if arithmetic_mean < 1e-10:
            return 0.0
            
        return geometric_mean / arithmetic_mean
    
    @staticmethod
    def compute_crest_factor(audio_window: np.ndarray) -> float:
        """
        Compute crest factor (peak amplitude / RMS ratio).
        
        Claps have higher crest factor (>5-6) due to sharp transient peak.
        Speech has lower crest factor (2-4 typical for voiced speech).
        
        Args:
            audio_window: Audio samples as numpy array
            
        Returns:
            Crest factor (dimensionless ratio)
        """
        peak = np.max(np.abs(audio_window))
        rms = np.sqrt(np.mean(audio_window ** 2))
        
        if rms < 1e-10:
            return float('inf')
            
        return peak / rms
    
    @staticmethod
    def compute_attack_time(audio_window: np.ndarray, sr: int) -> float:
        """
        Compute attack time (time to reach peak amplitude from start).
        
        Claps have very fast attack (<5ms typical).
        Speech has more gradual onset (>10ms typical).
        
        Args:
            audio_window: Audio samples as numpy array
            sr: Sample rate in Hz
            
        Returns:
            Attack time in milliseconds
        """
        envelope = np.abs(audio_window)
        peak_idx = np.argmax(envelope)
        
        if peak_idx == 0:
            return 0.0
            
        attack_samples = peak_idx
        attack_ms = attack_samples / sr * 1000
        return attack_ms
    
    @staticmethod
    def compute_zero_crossing_rate(audio_window: np.ndarray, sr: int) -> float:
        """
        Compute zero-crossing rate (number of sign changes per second).
        
        Claps have high initial ZCR due to broadband content.
        Speech has more moderate ZCR.
        
        Args:
            audio_window: Audio samples as numpy array
            sr: Sample rate in Hz
            
        Returns:
            Zero-crossing rate (crossings per second)
        """
        # Find sign changes
        signs = np.sign(audio_window)
        # Handle zeros by treating them as same sign as previous sample
        signs[signs == 0] = 1
        crossings = np.sum(np.abs(np.diff(signs))) > 0
        
        duration_seconds = len(audio_window) / sr
        if duration_seconds < 1e-10:
            return 0.0
            
        return crossings / duration_seconds
    
    @staticmethod
    def get_duration_ms(audio_window: np.ndarray, sr: int) -> float:
        """Get audio window duration in milliseconds."""
        return len(audio_window) / sr * 1000
    
    def extract_spectral_features(self, audio_window: np.ndarray) -> Dict[str, float]:
        """
        Extract all spectral features from an audio window.
        
        Args:
            audio_window: Audio samples as numpy array
            
        Returns:
            Dictionary containing all extracted spectral features
        """
        return {
            'spectral_centroid': self.compute_spectral_centroid(audio_window, self.sr),
            'hf_energy_ratio': self.compute_hf_energy_ratio(audio_window, self.sr),
            'spectral_flatness': self.compute_spectral_flatness(audio_window, self.sr),
        }
    
    def extract_temporal_features(self, audio_window: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features from an audio window.
        
        Args:
            audio_window: Audio samples as numpy array
            
        Returns:
            Dictionary containing all extracted temporal features
        """
        return {
            'duration_ms': self.get_duration_ms(audio_window, self.sr),
            'attack_time_ms': self.compute_attack_time(audio_window, self.sr),
            'crest_factor': self.compute_crest_factor(audio_window),
            'zcr': self.compute_zero_crossing_rate(audio_window, self.sr),
        }
    
    def classify_burst(
        self, 
        audio_window: np.ndarray,
        centroid_threshold: float = 3000.0,
        hf_ratio_threshold: float = 0.15,
        crest_threshold: float = 5.0,
        min_duration_ms: float = 10.0,
        max_duration_ms: float = 150.0,
        attack_time_threshold: float = 8.0,
        confidence_threshold: float = 2.5
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Classify an audio burst as clap or not using weighted feature scoring.
        
        Args:
            audio_window: Audio samples as numpy array
            centroid_threshold: Spectral centroid threshold in Hz
            hf_ratio_threshold: High-frequency energy ratio threshold
            crest_threshold: Crest factor threshold
            min_duration_ms: Minimum valid duration in ms
            max_duration_ms: Maximum valid duration in ms
            attack_time_threshold: Maximum attack time in ms
            
        Returns:
            Tuple of (is_clap, confidence_score, features_dict)
        """
        # Extract all features
        spectral_features = self.extract_spectral_features(audio_window)
        temporal_features = self.extract_temporal_features(audio_window)
        
        all_features = {**spectral_features, **temporal_features}
        
        # Compute weighted score
        score = 0.0
        
        # Spectral centroid check (higher is more clap-like)
        centroid = spectral_features['spectral_centroid']
        if centroid > centroid_threshold + 500:
            score += 1.0 * self.WEIGHT_CENTROID
        elif centroid > centroid_threshold:
            score += 0.7 * self.WEIGHT_CENTROID
        elif centroid > centroid_threshold - 500:
            score += 0.3 * self.WEIGHT_CENTROID
        
        # High-frequency energy ratio check (higher is more clap-like)
        hf_ratio = spectral_features['hf_energy_ratio']
        if hf_ratio > hf_ratio_threshold + 0.1:
            score += 1.0 * self.WEIGHT_HF_RATIO
        elif hf_ratio > hf_ratio_threshold:
            score += 0.7 * self.WEIGHT_HF_RATIO
        elif hf_ratio > hf_ratio_threshold - 0.05:
            score += 0.3 * self.WEIGHT_HF_RATIO
        
        # Crest factor check (higher is more clap-like)
        crest = temporal_features['crest_factor']
        if crest > crest_threshold + 2:
            score += 1.0 * self.WEIGHT_CREST
        elif crest > crest_threshold:
            score += 0.7 * self.WEIGHT_CREST
        elif crest > crest_threshold - 1:
            score += 0.3 * self.WEIGHT_CREST
        
        # Duration check (short bursts are more clap-like)
        duration_ms = temporal_features['duration_ms']
        if min_duration_ms <= duration_ms <= max_duration_ms * 0.7:
            score += 1.0 * self.WEIGHT_DURATION
        elif min_duration_ms <= duration_ms <= max_duration_ms:
            score += 0.6 * self.WEIGHT_DURATION
        
        # Attack time check (faster is more clap-like)
        attack_ms = temporal_features['attack_time_ms']
        if attack_ms < attack_time_threshold * 0.5:
            score += 1.0 * self.WEIGHT_ATTACK
        elif attack_ms < attack_time_threshold:
            score += 0.7 * self.WEIGHT_ATTACK
        elif attack_ms < attack_time_threshold + 3:
            score += 0.3 * self.WEIGHT_ATTACK
        
        is_clap = score >= confidence_threshold
        
        return is_clap, score, all_features
    
    @staticmethod
    def analyze_burst_for_debug(
        audio_window: np.ndarray,
        sr: int,
        detector: 'ClapDetector' = None
    ) -> Dict[str, any]:
        """
        Analyze a burst and return detailed feature information for debugging.
        
        Args:
            audio_window: Audio samples as numpy array
            sr: Sample rate in Hz
            detector: Optional ClapDetector instance (creates one if not provided)
            
        Returns:
            Dictionary with all features and analysis results
        """
        if detector is None:
            detector = ClapDetector(sr)
        
        spectral_features = detector.extract_spectral_features(audio_window)
        temporal_features = detector.extract_temporal_features(audio_window)
        
        # Compute RMS for reference
        rms = np.sqrt(np.mean(audio_window ** 2))
        peak = np.max(np.abs(audio_window))
        
        return {
            'spectral': spectral_features,
            'temporal': temporal_features,
            'energetic': {
                'rms': rms,
                'peak': peak,
            },
            'duration_samples': len(audio_window),
        }
