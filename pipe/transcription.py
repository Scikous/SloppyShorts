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
                "speaker": "UNKNOWN", 
                "tags": [] 
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
            
            found_speaker = "UNKNOWN"
            # Unpacking works for both tuples and lists (from JSON)
            for (d_start, d_end, label) in diarization_segments:
                if d_start <= seg_mid <= d_end:
                    found_speaker = label
                    break
            
            segment['speaker'] = found_speaker

    @staticmethod
    def save_index(index, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)

    @staticmethod
    def load_index(path) -> List[dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


class SubtitleGenerator:
    @staticmethod
    def generate_srt(master_index: List[dict], final_intervals: List[tuple], output_path: str):
        """
        Generates an SRT file for the 'Clean' video by shifting timestamps.
        final_intervals: List of (start, end) tuples from the original video that were kept.
        """
        print(f"-> Generating SRT for {output_path}...")
        
        srt_entries = []
        current_video_time = 0.0
        
        def format_timestamp(seconds):
            millis = int((seconds % 1) * 1000)
            seconds = int(seconds)
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

        entry_idx = 1
        final_intervals.sort(key=lambda x: x[0])
        
        # Optimization: Pointer for master_index
        seg_ptr = 0
        n_segs = len(master_index)
        
        for (chunk_start, chunk_end) in final_intervals:
            chunk_duration = chunk_end - chunk_start
            
            # 1. Advance seg_ptr to the first segment that could possibly overlap (end > chunk_start)
            while seg_ptr < n_segs and master_index[seg_ptr]['end'] <= chunk_start:
                seg_ptr += 1
            
            # 2. Iterate from seg_ptr
            temp_ptr = seg_ptr
            while temp_ptr < n_segs:
                seg = master_index[temp_ptr]
                
                # Optimization: If segment starts after this chunk ends, we can stop checking for this chunk
                if seg['start'] >= chunk_end:
                    break
                
                # Check Intersection
                start_overlap = max(chunk_start, seg['start'])
                end_overlap = min(chunk_end, seg['end'])
                
                if start_overlap < end_overlap:
                    # Calculate shifted times
                    rel_start = start_overlap - chunk_start
                    rel_end = end_overlap - chunk_start
                    
                    new_start = current_video_time + rel_start
                    new_end = current_video_time + rel_end
                    
                    # Avoid zero-duration subs
                    if new_end - new_start > 0.01:
                        srt_entries.append({
                            "index": entry_idx,
                            "start": format_timestamp(new_start),
                            "end": format_timestamp(new_end),
                            "text": seg['text'].strip()
                        })
                        entry_idx += 1
                
                temp_ptr += 1
            
            current_video_time += chunk_duration

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in srt_entries:
                f.write(f"{entry['index']}\n")
                f.write(f"{entry['start']} --> {entry['end']}\n")
                f.write(f"{entry['text']}\n\n")