import json
from typing import List
from pathlib import Path
from pipe.audio_core import TimeMapper

class MasterIndexer:
    @staticmethod
    def build_index(transcription_data: List[dict], time_mapper: TimeMapper) -> List[dict]:
        print("--- Generating Master Index ---")
        master_index =[]
        
        for seg in transcription_data:
            # Sentence segments CAN span gaps. Map bounds normally.
            raw_start = time_mapper.clean_to_raw(seg['start'])
            raw_end = time_mapper.clean_to_raw(seg['end'])
            
            words =[]
            for w in seg.get('words', []):
                clean_w_start = w['start']
                clean_w_end = w['end']
                
                # Words SHOULD NOT span gaps. Map both bounds using the offset of the word's midpoint.
                w_mid = (clean_w_start + clean_w_end) / 2.0
                w_offset = time_mapper.get_offset(w_mid)
                
                words.append({
                    "word": w['word'],
                    "start": clean_w_start + w_offset,
                    "end": clean_w_end + w_offset,
                    "probability": w.get('probability', 0.0)
                })
            
            master_index.append({
                "text": seg['text'],
                "start": raw_start,
                "end": raw_end,
                "words": words,
                "speaker": "UNKNOWN", 
                "tags":[] 
            })
            
        return master_index

    @staticmethod
    def inject_speakers(master_index: List[dict], diarization_segments: List[tuple]):
        print("--- Injecting Speaker IDs ---")
        # Diarization segments are already mapped to raw time, sort them to optimize
        diarization_segments.sort(key=lambda x: x[0])
        
        for segment in master_index:
            seg_start = segment['start']
            seg_end = segment['end']
            seg_mid = (seg_start + seg_end) / 2
            
            found_speaker = "UNKNOWN"
            for (d_start, d_end, label) in diarization_segments:
                if d_start <= seg_mid <= d_end:
                    found_speaker = label
                    break
            
            segment['speaker'] = found_speaker

    @classmethod
    def run(cls, transcription_data: List[dict], diarization_segments: List[tuple], time_mapper: TimeMapper, master_index_path: str) -> List[dict]:
        if Path(master_index_path).exists():
            print("-> Found cached Master Index.")
            with open(master_index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        master_index = cls.build_index(transcription_data, time_mapper)
        cls.inject_speakers(master_index, diarization_segments)
        
        with open(master_index_path, 'w', encoding='utf-8') as f:
            json.dump(master_index, f, indent=2)
            
        return master_index