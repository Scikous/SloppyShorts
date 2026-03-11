import torch
import numpy as np
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import json

from pipe.config import Config
from pipe.utils import FFmpegWrapper, cleanup_gpu
from pipe.clap_detector import ClapDetector


class RecapDetector:
    """
    Detects the recap section in a video by finding audio spikes (e.g., claps).
    
    Strategy:
    1. Scan from END of video backwards to find first loud audio spike
    2. Everything before spike = main content
    3. Everything after spike = raw recap section
    4. Run VAD on raw recap, keep only the LAST segment
    5. Final structure: main content + final recap segment
    """
    
    @staticmethod
    def find_spike_from_end(audio_path: str) -> float:
        """
        Detects the recap marker by finding claps using multi-feature spectral analysis.
        
        Strategy (user-specified):
        1. Find all silence periods in the last N seconds of audio
        2. Look for short bursts of sound BETWEEN silences (the clap)
        3. Use ClapDetector to classify each burst as clap or not
        4. Pattern: speech -> silence -> CLAP -> silence -> speech
        
        Args:
            audio_path: Path to raw audio file
            
        Returns:
            spike_time in seconds from start of video
            If no spike found, returns duration (entire video is main content)
        """
        print("--- Step: Detecting Clap via Multi-Feature Spectral Analysis ---")
        
        # Load audio as numpy array
        audio_np = FFmpegWrapper.extract_pcm_memory(audio_path, Config.SAMPLE_RATE_WHISPER)
        
        duration = len(audio_np) / Config.SAMPLE_RATE_WHISPER
        print(f"[Clap Detector] Audio duration: {duration:.2f}s")
        
        # Parameters from config
        max_search_sec = Config.CLAP_MAX_SEARCH_SEC
        search_start_time = max(0, duration - max_search_sec)
        search_start_sample = int(search_start_time * Config.SAMPLE_RATE_WHISPER)
        
        # Silence detection parameters
        silence_threshold = Config.CLAP_SILENCE_THRESHOLD
        min_silence_duration_ms = Config.CLAP_MIN_SILENCE_MS
        
        # Convert to samples
        silence_window_ms = 10  # Small windows for precise detection
        window_samples = int(silence_window_ms * Config.SAMPLE_RATE_WHISPER / 1000)
        
        # Extract search region
        search_region = audio_np[search_start_sample:]
        
        # Compute RMS in small windows
        num_windows = len(search_region) // window_samples
        rms_values = np.zeros(num_windows)
        for i in range(num_windows):
            window = search_region[i * window_samples:(i + 1) * window_samples]
            rms_values[i] = np.sqrt(np.mean(window ** 2))
        
        print(f"[Clap Detector] Search region: {search_start_time:.2f}s to {duration:.2f}s")
        print(f"[Clap Detector] Silence threshold: {silence_threshold}")
        print(f"[Clap Detector] Min silence duration: {min_silence_duration_ms}ms")
        
        # Find all silence periods (consecutive windows below threshold)
        silence_mask = rms_values < silence_threshold
        
        # Group consecutive silence windows into segments
        silence_segments = []  # List of (start_window, end_window)
        in_silence = False
        silence_start = None
        
        for i in range(len(silence_mask)):
            if silence_mask[i] and not in_silence:
                in_silence = True
                silence_start = i
            elif not silence_mask[i] and in_silence:
                in_silence = False
                # Check if this silence is long enough
                silence_duration = (i - silence_start) * window_samples / Config.SAMPLE_RATE_WHISPER * 1000
                if silence_duration >= min_silence_duration_ms:
                    silence_segments.append((silence_start, i))
        
        # Handle case where audio ends in silence
        if in_silence:
            silence_duration = (len(silence_mask) - silence_start) * window_samples / Config.SAMPLE_RATE_WHISPER * 1000
            if silence_duration >= min_silence_duration_ms:
                silence_segments.append((silence_start, len(silence_mask)))
        
        print(f"[Clap Detector] Found {len(silence_segments)} silence periods")
        
        # Now look for short audio bursts BETWEEN silence segments
        # Pattern: silence -> SHORT_BURST -> silence
        burst_candidates = []  # List of candidate dicts
        
        for i in range(len(silence_segments) - 1):
            prev_silence_end = silence_segments[i][1]
            next_silence_start = silence_segments[i + 1][0]
            
            # The gap between silences is a potential clap
            burst_start_window = prev_silence_end
            burst_end_window = next_silence_start
            
            if burst_end_window > burst_start_window:
                burst_samples_start = search_start_sample + burst_start_window * window_samples
                burst_samples_end = search_start_sample + burst_end_window * window_samples
                
                # Extract the actual burst audio from full audio array
                burst_audio = audio_np[burst_samples_start:burst_samples_end]
                
                burst_duration_ms = len(burst_audio) / Config.SAMPLE_RATE_WHISPER * 1000
                
                # Check if this burst is within valid clap duration range
                if (Config.CLAP_MIN_DURATION_MS <= burst_duration_ms <=
                    Config.CLAP_MAX_DURATION_MS):
                    
                    start_time = burst_samples_start / Config.SAMPLE_RATE_WHISPER
                    end_time = burst_samples_end / Config.SAMPLE_RATE_WHISPER
                    
                    burst_candidates.append({
                        'start': start_time,
                        'end': end_time,
                        'duration_ms': burst_duration_ms,
                        'audio': burst_audio,
                        'window_start': burst_start_window,
                        'window_end': burst_end_window
                    })
        
        print(f"[Clap Detector] Found {len(burst_candidates)} bursts between silences")
        
        if len(burst_candidates) == 0:
            print(f"[Clap Detector] No short audio bursts found between silences")
            del audio_np, rms_values, search_region
            cleanup_gpu()
            return duration
        
        # Use ClapDetector to classify each burst
        clap_detector = ClapDetector(sample_rate=Config.SAMPLE_RATE_WHISPER)
        
        classified_candidates = []
        print(f"\n[Clap Detector] Analyzing bursts with multi-feature spectral analysis:")
        
        for i, candidate in enumerate(burst_candidates):
            is_clap, score, features = clap_detector.classify_burst(
                candidate['audio'],
                centroid_threshold=Config.CLAP_SPECTRAL_CENTROID_THRESHOLD,
                hf_ratio_threshold=Config.CLAP_HF_RATIO_THRESHOLD,
                crest_threshold=Config.CLAP_CREST_FACTOR_THRESHOLD,
                min_duration_ms=Config.CLAP_MIN_DURATION_MS,
                max_duration_ms=Config.CLAP_MAX_DURATION_MS,
                attack_time_threshold=Config.CLAP_ATTACK_TIME_MS,
                confidence_threshold=Config.CLAP_CONFIDENCE_THRESHOLD
            )
            
            candidate['is_clap'] = is_clap
            candidate['score'] = score
            candidate['features'] = features
            
            classified_candidates.append(candidate)
            
            status = "✓ CLAP" if is_clap else "✗ not clap"
            print(f"    {i+1}. {candidate['start']:.2f}s: duration={candidate['duration_ms']:.0f}ms, "
                  f"score={score:.2f}, centroid={features['spectral_centroid']:.0f}Hz, "
                  f"crest={features['crest_factor']:.1f} -> {status}")
        
        # Filter to only actual claps
        clap_candidates = [c for c in classified_candidates if c['is_clap']]
        
        print(f"\n[Clap Detector] Found {len(clap_candidates)} valid clap(s)")
        
        if len(clap_candidates) == 0:
            print("[Clap Detector] No claps detected - falling back to keyword/position detection")
            del audio_np, rms_values, search_region
            cleanup_gpu()
            return duration
        
        # Pick the first clap (earliest in time within search region)
        best_clap = clap_candidates[0]
        
        spike_time = best_clap['start']
        
        print(f"\n[Clap Detector] Selected clap at: {spike_time:.2f}s")
        print(f"[Clap Detector] Score: {best_clap['score']:.2f}")
        print(f"[Clap Detector] Features: centroid={best_clap['features']['spectral_centroid']:.0f}Hz, "
              f"crest_factor={best_clap['features']['crest_factor']:.1f}, "
              f"hF_ratio={best_clap['features']['hf_energy_ratio']:.3f}")
        
        # Cleanup
        del audio_np, rms_values, search_region
        cleanup_gpu()
        
        return spike_time

    @staticmethod
    def extract_final_recap_segment(
        audio_path: str, 
        spike_time: float
    ) -> Optional[Tuple[float, float]]:
        """
        Extracts the final VAD segment from the raw recap section.
        
        Process:
        1. Load audio and slice from spike_time to end
        2. Run VAD on this region
        3. Return only the LAST segment (if any exist)
        
        Args:
            audio_path: Path to raw audio file
            spike_time: Timestamp where recap section begins
            
        Returns:
            Tuple of (start, end) times in original video coordinates
            None if no speech found after spike
        """
        from pipe.vad import VADProcessor
        from pipe.audio_core import AudioProcessor
        
        print(f"\n--- Step: Extracting Final Recap Segment ---")
        print(f"[Recap Extractor] Processing region: {spike_time:.2f}s to end")
        
        # Load full audio
        audio_np = FFmpegWrapper.extract_pcm_memory(audio_path, Config.SAMPLE_RATE_WHISPER)
        duration = len(audio_np) / Config.SAMPLE_RATE_WHISPER
        
        if spike_time >= duration - 0.1:
            print("[Recap Extractor] Spike at or near end, no recap section")
            del audio_np
            cleanup_gpu()
            return None
        
        # Slice the recap region (spike_time to end)
        start_sample = int(spike_time * Config.SAMPLE_RATE_WHISPER)
        recap_audio_np = audio_np[start_sample:]
        recap_tensor = torch.from_numpy(recap_audio_np)
        
        # Run VAD on recap region only
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                                       model='silero_vad', 
                                       onnx=False)
        (get_speech_ts, _, _, _, _) = utils
        
        speech_ts = get_speech_ts(
            recap_tensor, 
            model, 
            threshold=0.5, 
            min_speech_duration_ms=250, 
            min_silence_duration_ms=Config.MIN_SILENCE_MS, 
            return_seconds=True
        )
        
        # Convert to segments (in recap-local time)
        recap_segments = []
        post_padding = 0.2
        pre_padding = 0.1
        recap_max_dur = len(recap_audio_np) / Config.SAMPLE_RATE_WHISPER
        
        for ts in speech_ts:
            start = max(0.0, ts['start'] - pre_padding)
            end = min(recap_max_dur, ts['end'] + post_padding)
            
            if recap_segments and start < recap_segments[-1][1]:
                # Merge overlapping segments
                recap_segments[-1] = (recap_segments[-1][0], 
                                      max(recap_segments[-1][1], end))
            else:
                recap_segments.append((start, end))
        
        # Cleanup VAD model
        del audio_np, recap_audio_np, recap_tensor
        cleanup_gpu()
        
        if not recap_segments:
            print("[Recap Extractor] No speech segments found after spike")
            return None
        
        print(f"[Recap Extractor] Found {len(recap_segments)} VAD segment(s) in recap region")
        
        # Take only the LAST segment
        last_segment_local = recap_segments[-1]
        
        # Convert back to original video time coordinates
        final_start = spike_time + last_segment_local[0]
        final_end = spike_time + last_segment_local[1]
        
        print(f"[Recap Extractor] Final recap segment: {final_start:.2f}s - {final_end:.2f}s")
        print(f"[Recap Extractor] Duration: {final_end - final_start:.2f}s")
        
        return (final_start, final_end)
    
    @staticmethod
    def filter_main_segments(
        all_vad_segments: list, 
        spike_time: float
    ) -> list:
        """
        Filters VAD segments to keep only those before the spike.
        
        Args:
            all_vad_segments: List of (start, end) tuples from full VAD run
            spike_time: Timestamp where recap section begins
            
        Returns:
            Filtered list containing only main content segments
        """
        # Keep segments that end at or before spike time
        main_segments = [
            (s, e) for s, e in all_vad_segments 
            if e <= spike_time
        ]
        
        print(f"[Recap Extractor] Main content: {len(main_segments)} VAD segment(s)")
        return main_segments
    
    @classmethod
    def run(cls, audio_path: str) -> Tuple[float, Optional[Tuple[float, float]]]:
        """
        Runs the complete recap detection pipeline.
        
        Args:
            audio_path: Path to raw audio file
            
        Returns:
            Tuple of (spike_time, final_recap_segment)
            final_recap_segment is None if no spike or no speech after spike
        """
        # Step 1: Find spike from end
        spike_time = cls.find_spike_from_end(audio_path)
        
        # Step 2: Extract final recap segment (if spike found before end)
        final_recap_segment = None
        if spike_time < get_audio_duration(audio_path) - 0.5:
            final_recap_segment = cls.extract_final_recap_segment(audio_path, spike_time)
        
        return spike_time, final_recap_segment


def get_audio_duration(audio_path: str) -> float:
    """Helper to get audio duration in seconds."""
    import subprocess
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)
    ]
    return float(subprocess.check_output(cmd).strip())
