import os
from pathlib import Path
from typing import List
from pipe.config import Config, VerticalMode
from pipe.utils import get_video_duration, FFmpegWrapper, cleanup_gpu
from pipe.renderer import SubtitleRenderer
from pipe.audio_core import TimeMapper

class SubtitleGenerator:
    @staticmethod
    def generate_srt(master_index: List[dict], final_intervals: List[tuple], output_path: str):
        print(f"-> Generating SRT for {output_path}...")
        srt_entries =[]
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
        
        seg_ptr = 0
        n_segs = len(master_index)
        
        for (chunk_start, chunk_end) in final_intervals:
            chunk_duration = chunk_end - chunk_start
            
            while seg_ptr < n_segs and master_index[seg_ptr]['end'] <= chunk_start:
                seg_ptr += 1
            
            temp_ptr = seg_ptr
            while temp_ptr < n_segs:
                seg = master_index[temp_ptr]
                
                if seg['start'] >= chunk_end:
                    break
                
                start_overlap = max(chunk_start, seg['start'])
                end_overlap = min(chunk_end, seg['end'])
                
                if start_overlap < end_overlap:
                    rel_start = start_overlap - chunk_start
                    rel_end = end_overlap - chunk_start
                    
                    new_start = current_video_time + rel_start
                    new_end = current_video_time + rel_end
                    
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

class VideoRenderer:
    @staticmethod
    def render_recap(input_video: str, recap_segments: List[dict], master_index: List[dict]):
        if not recap_segments:
            return
            
        last_recap = recap_segments[-1]
        output_recap = Config.OUTPUT_DIR / f"{Path(input_video).stem}_recap_9_16.mp4"
        
        if output_recap.exists():
            print(f"-> Skipping Recap (Exists): {output_recap}")
            return
            
        print(f"\n--- Path 1.1: Rendering Recap ({last_recap['start']:.2f}s - end of video) ---")
        
        last_recap_idx = master_index.index(last_recap)
        recap_sequence = master_index[last_recap_idx:]
        total_video_duration = get_video_duration(input_video)
        
        ass_path = Config.TEMP_DIR / "recap.ass"
        SubtitleRenderer.create_hormozi_ass(
            recap_sequence, 
            str(ass_path), 
            is_vertical=True,
            start_offset=last_recap['start']
        )
        
        temp_trim = Config.TEMP_DIR / "temp_recap_raw.mp4"
        FFmpegWrapper.trim_segment(
            input_path=input_video,
            output_path=str(temp_trim),
            start=last_recap['start'],
            end=total_video_duration,
            desc="Extracting Recap",
            copy_streams=True
        )
        
        SubtitleRenderer.burn_subtitles(
            str(temp_trim), 
            str(ass_path), 
            str(output_recap), 
            mode=VerticalMode.BLUR_BG
        )
        print(f"-> Recap saved to {output_recap}")

    @staticmethod
    def render_clean_longform(input_video: str, time_mapper, keep_segments: List[tuple], recap_segments: List[dict], master_index: List[dict]):
        output_clean = Config.OUTPUT_DIR / f"{Path(input_video).stem}_clean_16_9.mp4"
        srt_path = output_clean.with_suffix(".srt")
        
        if output_clean.exists() and srt_path.exists():
            print(f"-> Skipping Clean Video & SRT (Exists): {output_clean}")
            return
            
        print(f"\n--- Path 1.2: Rendering Clean Long-form ---")
        recap_ranges = [(r['start'], r['end']) for r in recap_segments]
        final_intervals = TimeMapper.get_clean_intervals(keep_segments, recap_ranges)
        
        FFmpegWrapper.concat_video_segments(
            input_path=input_video,
            segments=final_intervals,
            output_path=str(output_clean),
            desc="Creating Clean Video"
        )
        print(f"-> Clean video saved to {output_clean}")
        
        SubtitleGenerator.generate_srt(master_index, final_intervals, str(srt_path))

    @staticmethod
    def render_highlights(input_video: str, highlight_segments: List[dict]):
        print(f"\n--- Processing {len(highlight_segments)} Highlights ---")
        for i, highlight in enumerate(highlight_segments):
            output_path = Config.OUTPUT_DIR / f"highlight_{i}.mp4"
            if output_path.exists():
                print(f"-> Skipping Highlight {i} (Exists)")
                continue
                
            print(f"-> Rendering Highlight {i+1}: {highlight['text'][:50]}...")
            
            ass_path = Config.TEMP_DIR / f"highlight_{i}.ass"
            SubtitleRenderer.create_hormozi_ass(
                [highlight], 
                str(ass_path), 
                is_vertical=True,
                start_offset=highlight['start']
            )
            
            temp_trim = Config.TEMP_DIR / f"temp_trim_{i}.mp4"
            FFmpegWrapper.trim_segment(
                input_path=input_video,
                output_path=str(temp_trim),
                start=highlight['start'],
                end=highlight['end'],
                desc=f"Extracting Highlight {i+1}",
                copy_streams=True
            )
            
            SubtitleRenderer.burn_subtitles(
                str(temp_trim), 
                str(ass_path), 
                str(output_path), 
                mode=VerticalMode.BLUR_BG
            )
            print(f"-> Highlight {i+1} saved to {output_path}")
            
            if temp_trim.exists():
                temp_trim.unlink()

    @classmethod
    def run(cls, input_video: str, master_index: List[dict], time_mapper, keep_segments: List[tuple]):
        print("--- Applying Logic Filters ---")
        recap_segments = []
        highlight_segments =[]
        
        for seg in master_index:
            text = seg['text'].lower()
            duration = seg['end'] - seg['start']
            
            if "recap" in text:
                if 'recap' not in seg['tags']:
                    seg['tags'].append('recap')
                recap_segments.append(seg)
                
            if 5.0 < duration < 60.0:
                score = 1 
                if score >= 1:
                    if 'highlight' not in seg['tags']:
                        seg['tags'].append('highlight')
                    highlight_segments.append(seg)
                    
        cleanup_gpu()

        cls.render_recap(input_video, recap_segments, master_index)
        cls.render_clean_longform(input_video, time_mapper, keep_segments, recap_segments, master_index)
        cls.render_highlights(input_video, highlight_segments)