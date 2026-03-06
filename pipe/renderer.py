import pysubs2
from pysubs2 import SSAEvent, SSAStyle, Color
from typing import List, Tuple
from pipe.config import VerticalMode
from pipe.utils import FFmpegWrapper, get_video_duration

class SubtitleRenderer:
    @staticmethod
    def create_hormozi_ass(master_index_segments: List[dict], output_file: str, is_vertical: bool = False, start_offset: float = 0.0):
        """
        Generates .ass subtitles from the Master Index segments.
        
        Args:
            master_index_segments: List of segment dictionaries containing word timing data
            output_file: Path to save the .ass subtitle file
            is_vertical: Whether to use vertical (9:16) formatting
            start_offset: Time offset to subtract from all timestamps (for trimmed clips)
        """
        print(f"--- Generating ASS Subtitles ---")
        subs = pysubs2.SSAFile()
        
        if is_vertical:
            subs.info["PlayResX"] = "1080"
            subs.info["PlayResY"] = "1920"
            margin_v = 400
            font_size = 90
        else:
            subs.info["PlayResX"] = "1920"
            subs.info["PlayResY"] = "1080"
            margin_v = 100
            font_size = 80

        style = SSAStyle(
            fontname="Impact",
            fontsize=font_size,
            primarycolor=Color(255, 255, 0),
            secondarycolor=Color(0, 0, 0),
            outlinecolor=Color(0, 0, 0),
            backcolor=Color(0, 0, 0, 0),
            bold=True,
            alignment=5,
            outline=3,
            shadow=0,
            marginv=margin_v 
        )
        subs.styles["Hormozi"] = style

        all_words = []
        for seg in master_index_segments:
            if 'words' in seg:
                all_words.extend(seg['words'])

        # Create ONE subtitle event per word to avoid stacking/overlapping subtitles
        # Each word appears when it's spoken and disappears when the next word starts
        for i, word_data in enumerate(all_words):
            start_ms = int((word_data['start'] - start_offset) * 1000)
            
            # End time is either this word's end, or the start of the next word (whichever comes first)
            if i + 1 < len(all_words):
                end_ms = min(
                    int((word_data['end'] - start_offset) * 1000),
                    int((all_words[i + 1]['start'] - start_offset) * 1000)
                )
            else:
                # Last word - use its actual end time
                end_ms = int((word_data['end'] - start_offset) * 1000)
            
            text_content = word_data['word'].strip().upper()
            
            subs.events.append(SSAEvent(start=start_ms, end=end_ms, text=text_content, style="Hormozi"))

        subs.save(str(output_file))

    @staticmethod
    def adjust_timestamps_for_concat(ass_path: str, concat_segments: List[tuple], raw_start_offset: float):
        subs = pysubs2.load(ass_path)
        
        concat_map =[]  
        current_concat_time = 0.0
        
        for seg_start, seg_end in sorted(concat_segments):
            duration = seg_end - seg_start
            concat_map.append({
                'raw_start': seg_start,
                'raw_end': seg_end,
                'concat_start': current_concat_time,
            })
            current_concat_time += duration
            
        def map_time(raw_t):
            # 1. Exact match inside a segment
            for seg in concat_map:
                if seg['raw_start'] <= raw_t <= seg['raw_end']:
                    return (raw_t - seg['raw_start']) + seg['concat_start']
            
            # 2. Fallback snap to nearest segment boundary (if caught in a gap)
            closest_t = 0.0
            min_diff = float('inf')
            for seg in concat_map:
                dist_start = abs(raw_t - seg['raw_start'])
                if dist_start < min_diff:
                    min_diff = dist_start
                    closest_t = seg['concat_start']
                
                dist_end = abs(raw_t - seg['raw_end'])
                if dist_end < min_diff:
                    min_diff = dist_end
                    closest_t = seg['concat_start'] + (seg['raw_end'] - seg['raw_start'])
            return closest_t
        
        # Adjust each subtitle event's timestamps
        for event in subs:
            raw_event_start = event.start / 1000.0 + raw_start_offset
            raw_event_end = event.end / 1000.0 + raw_start_offset
            
            new_concat_start = map_time(raw_event_start)
            new_concat_end = map_time(raw_event_end)
            
            # Ensure minimum duration so subs don't vanish if snapped to the same boundary
            if new_concat_end <= new_concat_start:
                new_concat_end = new_concat_start + 0.05
            
            event.start = int(new_concat_start * 1000)
            event.end = int(new_concat_end * 1000)
        
        subs.save(str(ass_path))

    @staticmethod
    def burn_subtitles(
        video_path: str, 
        ass_path: str, 
        output_path: str, 
        mode: VerticalMode = VerticalMode.CROP_CENTER,
        facecam_coords: Tuple[int, int, int, int] = None
    ):
        print(f"--- Burning Subtitles (Mode: {mode.name}) ---")
        
        total_duration = get_video_duration(video_path)
        
        if mode == VerticalMode.CROP_CENTER:
            vf_chain = "crop=ih*(9/16):ih,scale=1080:1920"
            
        elif mode == VerticalMode.BLUR_BG:
            vf_chain = (
                "split=2[bg][fg];"
                "[bg]format=yuv420p,scale=iw/4:ih/4,boxblur=10,scale=1080:1920:flags=bilinear[bg_blurred];"
                "[fg]scale=1080:-1[fg_scaled];"
                "[bg_blurred][fg_scaled]overlay=(W-w)/2:(H-h)/2"
            )
            
        elif mode == VerticalMode.SPLIT_SCREEN:
            if not facecam_coords:
                raise ValueError("Facecam coordinates required for Split Screen.")
            fx, fy, fw, fh = facecam_coords
            
            vf_chain = (
                f"split=2[main][face];"
                f"[main]scale=1080:960:force_original_aspect_ratio=decrease[main_v];"
                f"[face]crop={fw}:{fh}:{fx}:{fy},scale=1080:960:force_original_aspect_ratio=decrease[face_v];"
                f"color=s=1080x1920:c=black[canvas];"
                f"[canvas][main_v]overlay=(W-w)/2:(960-h)/2[p1];"
                f"[p1][face_v]overlay=(W-w)/2:960+(960-h)/2"
            )
        else:
            vf_chain = "null"

        # Escape the path for FFmpeg filter
        escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
        
        if vf_chain == "null":
            full_vf = f"ass='{escaped_ass}', setsar=1"
        else:
            full_vf = f"{vf_chain}[v_final];[v_final]ass='{escaped_ass}', setsar=1"

        FFmpegWrapper.render_with_filter(
            input_path=video_path,
            output_path=output_path,
            filter_complex=full_vf,
            duration=total_duration,
            desc=f"Burning ({mode.name})"
        )