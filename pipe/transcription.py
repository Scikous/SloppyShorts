import json
from faster_whisper import WhisperModel
from pipe.config import Config
from pipe.utils import cleanup_gpu
from pipe.audio_core import TimeMapper
from typing import List

class MasterIndexer:
    @staticmethod
    def create_index(clean_audio_path: str, time_mapper: TimeMapper) -> List[dict]:
        print("--- Generating Master Index ---")
        model = WhisperModel(Config.WHISPER_MODEL, device=Config.DEVICE, compute_type=Config.COMPUTE_TYPE)
        
        segments, _ = model.transcribe(
            str(clean_audio_path),
            beam_size=5,
            word_timestamps=True,
            vad_filter=False # We already did VAD
        )
        
        master_index = []
        
        for seg in segments:
            # Map Segment Times
            raw_start = time_mapper.clean_to_raw(seg.start)
            raw_end = time_mapper.clean_to_raw(seg.end)
            
            # Map Word Times
            words = []
            if seg.words:
                for w in seg.words:
                    w_raw_start = time_mapper.clean_to_raw(w.start)
                    w_raw_end = time_mapper.clean_to_raw(w.end)
                    words.append({
                        "word": w.word,
                        "start": w_raw_start,
                        "end": w_raw_end,
                        "probability": w.probability
                    })
            
            master_index.append({
                "text": seg.text,
                "start": raw_start,
                "end": raw_end,
                "words": words,
                "speaker": "UNKNOWN", # Placeholder for Diarization update
                "tags": [] # 'recap', 'highlight', etc.
            })
            
        cleanup_gpu()
        return master_index


    @staticmethod
    def inject_speakers(master_index: List[dict], diarization_segments: List[tuple]):
        """
        Matches transcript segments to speaker labels based on timestamp overlap.
        diarization_segments: [(start, end, speaker_label), ...]
        """
        print("--- Injecting Speaker IDs ---")
        # Sort both lists by start time to optimize matching
        diarization_segments.sort(key=lambda x: x[0])
        
        for segment in master_index:
            seg_start = segment['start']
            seg_end = segment['end']
            seg_mid = (seg_start + seg_end) / 2
            
            # Simple logic: Find the speaker active at the midpoint of the sentence
            # (A more robust version calculates % overlap, but this suffices for MVP)
            found_speaker = "UNKNOWN"
            for (d_start, d_end, label) in diarization_segments:
                if d_start <= seg_mid <= d_end:
                    found_speaker = label
                    break
            
            segment['speaker'] = found_speaker


    @staticmethod
    def save_index(index, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)