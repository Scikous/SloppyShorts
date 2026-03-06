import torch
import subprocess
from typing import List, Tuple
from pipe.config import Config
from pipe.utils import FFmpegWrapper

class TimeMapper:
    """
    The Mathematical Anchor.
    Converts 'Clean Audio Time' (what Whisper/Pyannote sees) back to 'Raw Video Time'.
    """
    def __init__(self, keep_segments: List[Tuple[float, float]]):
        self.segments = sorted(keep_segments, key=lambda x: x[0])
        self.map =[]
        
        current_clean_cursor = 0.0
        for start, end in self.segments:
            duration = end - start
            self.map.append({
                "raw_start": start,
                "raw_end": end,
                "clean_start": current_clean_cursor,
                "clean_end": current_clean_cursor + duration,
                "offset": start - current_clean_cursor 
            })
            current_clean_cursor += duration

    def get_offset(self, clean_time: float) -> float:
        """Determines the correct temporal offset to apply for a given clean timestamp."""
        if not self.map: 
            return 0.0
            
        for seg in self.map:
            # Use strict less-than for the end boundary to prevent boundary overlaps
            if seg["clean_start"] <= clean_time < seg["clean_end"]:
                return seg["offset"]
                
        # Handle exact end of the very last segment
        if clean_time == self.map[-1]["clean_end"]:
            return self.map[-1]["offset"]
            
        # Fallback handling
        if clean_time < self.map[0]["clean_start"]:
            return self.map[0]["offset"]
        if clean_time > self.map[-1]["clean_end"]:
            return self.map[-1]["offset"]
            
        return 0.0

    def clean_to_raw(self, clean_time: float) -> float:
        """Converts a timestamp from the silence-removed audio back to original video time."""
        return clean_time + self.get_offset(clean_time)

    @staticmethod
    def get_clean_intervals(keep_segments: List[Tuple[float, float]], drop_ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Subtracts drop_ranges (e.g. Recaps) from the keep_segments."""
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


class AudioProcessor:
    @staticmethod
    def extract_raw_audio(input_video: str, output_wav: str, sr: int = 16000):
        cmd =[
            "ffmpeg", "-y", "-v", "error",
            "-i", str(input_video),
            "-vn", "-ac", "1", "-ar", str(sr),
            str(output_wav)
        ]
        subprocess.run(cmd, check=True)

    @staticmethod
    def load_audio(path: str, sr: int = 16000) -> torch.Tensor:
        audio_np = FFmpegWrapper.extract_pcm_memory(path, sr)
        return torch.from_numpy(audio_np)

    @staticmethod
    def create_clean_audio(input_video: str, segments: List[Tuple[float, float]], output_wav: str):
        FFmpegWrapper.concat_audio_segments(
            input_video, 
            segments, 
            output_wav, 
            Config.SAMPLE_RATE_WHISPER
        )