import os
from pathlib import Path
from typing import List, Optional, Tuple
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
    def render_recap(input_video: str, recap_segments: List[dict], master_index: List[dict], time_mapper, keep_segments: List[tuple], final_recap_segment: Optional[Tuple[float, float]] = None):
        if not recap_segments:
            return
            
        last_recap = recap_segments[-1]
        output_recap = Config.OUTPUT_DIR / f"{Path(input_video).stem}_recap_9_16.mp4"
        
        if output_recap.exists():
            print(f"-> Skipping Recap (Exists): {output_recap}")
            return
        
        # FIX: Use final_recap_segment times if provided, otherwise fall back to last_recap from master_index
        # This ensures we use the actual detected clap/recap times instead of VAD segment times
        if final_recap_segment:
            recap_start = final_recap_segment[0]
            recap_end = final_recap_segment[1]
            print(f"\n--- Path 1.1: Rendering Recap using final_recap_segment ({recap_start:.2f}s - {recap_end:.2f}s) ---")
        else:
            recap_start = last_recap['start']
            recap_end = last_recap['end']
            print(f"\n--- Path 1.1: Rendering Recap using master_index ({recap_start:.2f}s - end of video) ---")
        
        # Filter master_index segments to only include those whose END time is >= recap_start
        # This prevents subtitles from before the clap from appearing at the start of the recap video
        recap_sequence = [seg for seg in master_index if seg['end'] >= recap_start and seg['start'] < recap_end]
        
        # Further filter words within each segment to exclude those ending before recap_start
        filtered_recap_sequence = []
        for seg in recap_sequence:
            if 'words' in seg:
                # Only keep words that start after recap_start (or overlap with it)
                valid_words = [w for w in seg['words'] if w['start'] >= recap_start - 0.1]  # Small tolerance for timing variations
                if valid_words:
                    filtered_seg = seg.copy()
                    filtered_seg['words'] = valid_words
                    filtered_recap_sequence.append(filtered_seg)
            else:
                filtered_recap_sequence.append(seg)
        
        recap_sequence = filtered_recap_sequence
        
        total_video_duration = get_video_duration(input_video)
        
        # Filter keep_segments to only include portions from recap_start onwards
        recap_keep_segments = []
        for seg_start, seg_end in keep_segments:
            if seg_end <= recap_start:
                continue  # Segment is entirely before recap
            clipped_start = max(seg_start, recap_start)
            clipped_end = min(seg_end, total_video_duration)
            if clipped_start < clipped_end:
                recap_keep_segments.append((clipped_start, clipped_end))
        
        print(f"[INFO] Filtered {len(recap_keep_segments)} keep_segments for recap: {[f'{s:.2f}-{e:.2f}' for s, e in recap_keep_segments]}")
        
        # Generate ASS with adjusted timestamps (relative to the concatenated video timeline)
        ass_path = Config.TEMP_DIR / "recap.ass"
        SubtitleRenderer.create_hormozi_ass(
            recap_sequence,
            str(ass_path),
            is_vertical=True,
            start_offset=recap_start
        )
        
        # Use concat_video_segments to remove silence gaps
        temp_concat = Config.TEMP_DIR / "temp_recap_concat.mp4"
        FFmpegWrapper.concat_video_segments(
            input_path=input_video,
            segments=recap_keep_segments,
            output_path=str(temp_concat),
            desc="Creating Recap (silence removed)"
        )
        
        # Adjust subtitle timestamps to match the concatenated timeline
        SubtitleRenderer.adjust_timestamps_for_concat(
            str(ass_path),
            recap_keep_segments,
            recap_start
        )
        
        SubtitleRenderer.burn_subtitles(
            str(temp_concat),
            str(ass_path),
            str(output_recap),
            mode=VerticalMode.BLUR_BG
        )
        print(f"-> Recap saved to {output_recap}")
        
        if temp_concat.exists():
            temp_concat.unlink()

    @staticmethod
    def render_clean_longform(input_video: str, time_mapper, keep_segments: List[tuple], recap_segments: List[dict], master_index: List[dict]):
        output_clean = Config.OUTPUT_DIR / f"{Path(input_video).stem}_clean_16_9.mp4"
        srt_path = output_clean.with_suffix(".srt")
        
        if output_clean.exists() and srt_path.exists():
            print(f"-> Skipping Clean Video & SRT (Exists): {output_clean}")
            return
            
        print(f"\n--- Path 1.2: Rendering Clean Long-form ---")
        # FIX: Use keep_segments directly - it already contains main content + final recap segment
        # No need to subtract recap_ranges since we WANT the recap in the clean video
        final_intervals = keep_segments
        
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
    def run(cls, input_video: str, master_index: List[dict], time_mapper, keep_segments: List[tuple], final_recap_segment: Optional[Tuple[float, float]] = None):
        """Main entry point for rendering."""
        print("--- Applying Logic Filters ---")
        
        # Filter segments based on final_recap_segment (if provided)
        if final_recap_segment:
            recap_start, recap_end = final_recap_segment
            # Find master_index segments that OVERLAP with the final recap segment
            # Use overlap detection instead of strict containment to handle timestamp variations
            # from VAD padding and TimeMapper offset calculations
            recap_segments = [
                seg for seg in master_index
                if seg['start'] < recap_end and seg['end'] > recap_start
            ]
            print(f"[INFO] Found {len(recap_segments)} segments overlapping final recap section (time: {recap_start:.2f}s - {recap_end:.2f}s)")
        else:
            # Fallback to keyword-based detection if no recap segment provided
            recap_segments = []
            for seg in master_index:
                text = seg['text'].lower()
                if "recap" in text:
                    if 'recap' not in seg['tags']:
                        seg['tags'].append('recap')
                    recap_segments.append(seg)
            print(f"[INFO] Using keyword-based detection: {len(recap_segments)} recap segments found")
        
        # Detect highlights based on duration
        highlight_segments = []
        for seg in master_index:
            duration = seg['end'] - seg['start']
            if 5.0 < duration < 60.0:
                score = 1
                if score >= 1:
                    if 'highlight' not in seg['tags']:
                        seg['tags'].append('highlight')
                    highlight_segments.append(seg)
                    
        cleanup_gpu()

        # Pass final_recap_segment to render_recap so it uses actual detected times
        cls.render_recap(input_video, recap_segments, master_index, time_mapper, keep_segments, final_recap_segment)
        cls.render_clean_longform(input_video, time_mapper, keep_segments, recap_segments, master_index)
        cls.render_highlights(input_video, highlight_segments)
