import json
import torch
from typing import List, Tuple
from pathlib import Path
from pyannote.audio import Pipeline
from pipe.utils import cleanup_gpu
from pipe.audio_core import TimeMapper

class Diarizer:
    @staticmethod
    def process(audio_path: str, hf_token: str) -> List[Tuple[float, float, str]]:
        print(f"--- Step: Speaker Diarization ---")
        print(f"[PyAnnote] Loading pipeline...")
        
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        print(f"[Diarizer] Running inference...")
        result = pipeline(str(audio_path))

        if hasattr(result, "speaker_diarization"):
            annotation = result.speaker_diarization
        else:
            annotation = result

        segments =[]
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        
        cleanup_gpu()
        return segments

    @classmethod
    def run(cls, clean_audio_path: str, diarization_path: str, hf_token: str, time_mapper: TimeMapper = None) -> List[Tuple[float, float, str]]:
        if Path(diarization_path).exists():
            print("-> Found cached Diarization.")
            with open(diarization_path, 'r') as f:
                return [tuple(x) for x in json.load(f)]
                
        print("-> Running Diarization on clean audio...")
        segments_clean = cls.process(str(clean_audio_path), hf_token)
        
        # Map clean-time diarization segments back to raw video time immediately
        segments_raw =[]
        if time_mapper:
            for start, end, speaker in segments_clean:
                raw_start = time_mapper.clean_to_raw(start)
                raw_end = time_mapper.clean_to_raw(end)
                segments_raw.append((raw_start, raw_end, speaker))
        else:
            segments_raw = segments_clean
            
        with open(diarization_path, 'w') as f:
            json.dump(segments_raw, f)
            
        return segments_raw