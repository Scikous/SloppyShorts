import torch
import numpy as np
import subprocess
import soundfile as sf
from typing import List, Dict, Tuple
from pipe.config import Config
from pipe.utils import cleanup_gpu


# --- CRITICAL FIX FOR IMPORT CRASH ---
# We must configure torchaudio BEFORE importing pyannote.
# import torchaudio
# try:
#     # Force the 'soundfile' backend. 
#     # The default backend often triggers C++ vector errors when mixed with system FFmpeg.
#     torchaudio.set_audio_backend("soundfile")
# except Exception:
#     pass
# # -------------------------------------


class TimeMapper:
    """
    The Mathematical Anchor.
    Converts 'Clean Audio Time' (what Whisper sees) back to 'Raw Video Time'.
    """
    def __init__(self, keep_segments: List[Tuple[float, float]]):
        # keep_segments = [(start, end), (start, end)] in Raw Time
        self.segments = sorted(keep_segments, key=lambda x: x[0]) #maybe better solution than sorted?
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
        return -1.0 # Should not happen if logic is tight


class Diarizer:
    @staticmethod
    def process(audio_path: str, hf_token) -> List[Tuple[float, float, str]]:
        """
        Runs Pyannote on the RAW audio to identify speakers.
        Returns: [(start, end, speaker_label), ...]
        """
        print(f"--- Step: Speaker Diarization ---")
        from pyannote.audio import Pipeline
        print("WAHHAHAHAHAHAHHAHAHAHAH")
        print(f"[PyAnnote] Loading pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            token=hf_token
        )
        
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        print(f"[PyAnnote] Processing ")#{audio_path}...")
        # Run inference
        # result = pipeline(str(audio_path))
        print(f"[Diarizer] Running inference...")
        inputs = {"waveform": audio_path.unsqueeze(0), "sample_rate": 16000}
        result = pipeline(inputs)
        # result = pipeline(audio_path)

        # --- FIX FOR PYANNOTE 4.x vs 3.x ---
        if hasattr(result, "speaker_diarization"):
            annotation = result.speaker_diarization
        else:
            annotation = result

        segments = []
        # annotation.itertracks(yield_label=True) yields (Segment, TrackID, Label)
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        
        cleanup_gpu()
        return segments

class AudioProcessor:
    @staticmethod
    def load_audio(path: str, sr: int = 16000) -> torch.Tensor:
        cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
        process = subprocess.run(cmd, capture_output=True, check=True)
        audio_np = np.frombuffer(process.stdout, np.float32).copy()
        return torch.from_numpy(audio_np)

    @staticmethod
    def get_vad_segments(audio_tensor: torch.Tensor) -> List[Tuple[float, float]]:
        """Returns list of (start, end) timestamps to KEEP."""
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=False)
        (get_speech_ts, _, _, _, _) = utils
        
        # Get speech timestamps
        speech_ts = get_speech_ts(
            audio_tensor, 
            model, 
            threshold=0.5, 
            min_speech_duration_ms=250, 
            min_silence_duration_ms=Config.MIN_SILENCE_MS, 
            return_seconds=True
        )
        
        # Convert dict to list of tuples and add padding
        segments = []
        padding = 0.2 # 200ms padding
        max_dur = len(audio_tensor) / Config.SAMPLE_RATE_WHISPER
        
        for ts in speech_ts:
            start = max(0, ts['start'] - padding)
            end = min(max_dur, ts['end'] + padding)
            # Merge overlaps
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
        Uses FFmpeg filter_complex to avoid re-encoding video.
        """
        filter_parts = []
        for i, (start, end) in enumerate(segments):
            filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];")
        
        concat_filter = "".join(filter_parts) + f"{''.join([f'[a{i}]' for i in range(len(segments))])}concat=n={len(segments)}:v=0:a=1[outa]"
        
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(input_video),
            "-filter_complex", concat_filter,
            "-map", "[outa]",
            "-ar", str(Config.SAMPLE_RATE_WHISPER),
            str(output_wav)
        ]
        subprocess.run(cmd, check=True)