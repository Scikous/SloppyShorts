import torch
import subprocess
import json
from typing import List, Tuple
from pipe.config import Config
from pipe.utils import cleanup_gpu, FFmpegWrapper

class TimeMapper:
    """
    The Mathematical Anchor.
    Converts 'Clean Audio Time' (what Whisper sees) back to 'Raw Video Time'.
    """
    def __init__(self, keep_segments: List[Tuple[float, float]]):
        # keep_segments = [(start, end), (start, end)] in Raw Time
        self.segments = sorted(keep_segments, key=lambda x: x[0])
        self.map = []
        
        current_clean_cursor = 0.0
        for start, end in self.segments:
            duration = end - start
            self.map.append({
                "raw_start": start,
                "raw_end": end,
                "clean_start": current_clean_cursor,
                "clean_end": current_clean_cursor + duration,
                "offset": start - current_clean_cursor # Add this to Clean Time to get Raw Time
            })
            current_clean_cursor += duration

    def clean_to_raw(self, clean_time: float) -> float:
        """Converts a timestamp from the silence-removed audio back to original video time."""
        for seg in self.map:
            if seg["clean_start"] <= clean_time <= seg["clean_end"]:
                return clean_time + seg["offset"]
        return -1.0 

    @staticmethod
    def get_clean_intervals(keep_segments: List[Tuple[float, float]], drop_ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Subtracts 'drop_ranges' (Recaps) from 'keep_segments' (VAD non-silence).
        Returns a new list of start/end times to keep for the final long-form video.
        """
        final_intervals = []
        drop_ranges = sorted(drop_ranges, key=lambda x: x[0])
        
        for k_start, k_end in keep_segments:
            current_start = k_start
            
            for d_start, d_end in drop_ranges:
                if d_end <= current_start or d_start >= k_end:
                    continue 
                
                if d_start > current_start:
                    final_intervals.append((current_start, d_start))
                
                current_start = max(current_start, d_end)
                
            if current_start < k_end:
                final_intervals.append((current_start, k_end))
                
        return final_intervals


class Diarizer:
    @staticmethod
    def process(audio_path: str, hf_token) -> List[Tuple[float, float, str]]:
        """
        Runs Pyannote on the RAW audio to identify speakers.
        Returns: [(start, end, speaker_label), ...]
        """
        print(f"--- Step: Speaker Diarization ---")
        from pyannote.audio import Pipeline
        
        print(f"[PyAnnote] Loading pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            token=hf_token
        )
        
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        print(f"[Diarizer] Running inference...")
        # Passing path is safer for memory than passing tensor
        result = pipeline(str(audio_path))

        if hasattr(result, "speaker_diarization"):
            annotation = result.speaker_diarization
        else:
            annotation = result

        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        
        cleanup_gpu()
        return segments

class AudioProcessor:
    @staticmethod
    def extract_raw_audio(input_video: str, output_wav: str, sr: int = 16000):
        """Extracts mono audio from video to WAV file."""
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(input_video),
            "-vn", "-ac", "1", "-ar", str(sr),
            str(output_wav)
        ]
        subprocess.run(cmd, check=True)

    @staticmethod
    def load_audio(path: str, sr: int = 16000) -> torch.Tensor:
        """Loads audio using FFmpegWrapper into a Torch Tensor."""
        audio_np = FFmpegWrapper.extract_pcm_memory(path, sr)
        return torch.from_numpy(audio_np)

    @staticmethod
    def get_vad_segments(audio_tensor: torch.Tensor) -> List[Tuple[float, float]]:
        """Returns list of (start, end) timestamps to KEEP."""
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=False)
        (get_speech_ts, _, _, _, _) = utils
        
        speech_ts = get_speech_ts(
            audio_tensor, 
            model, 
            threshold=0.5, 
            min_speech_duration_ms=250, 
            min_silence_duration_ms=Config.MIN_SILENCE_MS, 
            return_seconds=True
        )
        
        segments = []
        padding = 0.2 
        max_dur = len(audio_tensor) / Config.SAMPLE_RATE_WHISPER
        
        for ts in speech_ts:
            start = max(0, ts['start'] - padding)
            end = min(max_dur, ts['end'] + padding)
            if segments and start < segments[-1][1]:
                segments[-1] = (segments[-1][0], max(segments[-1][1], end))
            else:
                segments.append((start, end))
                
        cleanup_gpu()
        return segments

    @staticmethod
    def create_clean_audio(input_video: str, segments: List[Tuple[float, float]], output_wav: str):
        """
        Creates a temporary WAV file with silence removed for Whisper.
        Delegates to FFmpegWrapper.
        """
        FFmpegWrapper.concat_audio_segments(
            input_video, 
            segments, 
            output_wav, 
            Config.SAMPLE_RATE_WHISPER
        )

    @staticmethod
    def save_segments(segments: List[Tuple[float, float]], path: str):
        with open(path, 'w') as f:
            json.dump(segments, f)

    @staticmethod
    def load_segments(path: str) -> List[Tuple[float, float]]:
        with open(path, 'r') as f:
            # JSON loads as lists, convert to tuples if preferred, though lists work for iteration
            return [tuple(x) for x in json.load(f)]