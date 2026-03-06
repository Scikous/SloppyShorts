import torch
import json
from typing import List, Tuple
from pathlib import Path
from pipe.config import Config
from pipe.utils import cleanup_gpu
from pipe.audio_core import AudioProcessor

class VADProcessor:
    @staticmethod
    def get_vad_segments(audio_tensor: torch.Tensor) -> List[Tuple[float, float]]:
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
        
        segments =[]
        post_padding = 0.2  
        pre_padding = 0.1 # <--- ADD 100ms padding BEFORE speech to anchor timestamps context
        max_dur = len(audio_tensor) / Config.SAMPLE_RATE_WHISPER
        
        for ts in speech_ts:
            start = max(0.0, ts['start'] - pre_padding) 
            end = min(max_dur, ts['end'] + post_padding)
            if segments and start < segments[-1][1]:
                segments[-1] = (segments[-1][0], max(segments[-1][1], end))
            else:
                segments.append((start, end))
                
        cleanup_gpu()
        return segments

    @staticmethod
    def save_segments(segments: List[Tuple[float, float]], path: str):
        with open(path, 'w') as f:
            json.dump(segments, f)

    @staticmethod
    def load_segments(path: str) -> List[Tuple[float, float]]:
        with open(path, 'r') as f:
            return [tuple(x) for x in json.load(f)]
            
    @classmethod
    def run(cls, raw_audio_path: str, vad_segments_path: str) -> List[Tuple[float, float]]:
        if Path(vad_segments_path).exists():
            print("-> Found cached VAD segments.")
            return cls.load_segments(vad_segments_path)
            
        print("-> Running VAD...")
        wav_tensor = AudioProcessor.load_audio(raw_audio_path)
        keep_segments = cls.get_vad_segments(wav_tensor)
        cls.save_segments(keep_segments, vad_segments_path)
        
        del wav_tensor
        cleanup_gpu()
        return keep_segments