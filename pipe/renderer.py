# import pysubs2
# from pysubs2 import SSAEvent, SSAStyle, Color
# from typing import List, Tuple
# from pipe.config import VerticalMode
# from pipe.utils import FFmpegWrapper, get_video_duration

# class SubtitleRenderer:
#     @staticmethod
#     def create_hormozi_ass(master_index_segments: List[dict], output_file: str, is_vertical: bool = False):
#         """
#         Generates .ass subtitles from the Master Index segments.
#         """
#         print(f"--- Generating ASS Subtitles ---")
#         subs = pysubs2.SSAFile()
        
#         if is_vertical:
#             subs.info["PlayResX"] = "1080"
#             subs.info["PlayResY"] = "1920"
#             margin_v = 400
#             font_size = 90
#         else:
#             subs.info["PlayResX"] = "1920"
#             subs.info["PlayResY"] = "1080"
#             margin_v = 100
#             font_size = 80

#         style = SSAStyle(
#             fontname="Impact",
#             fontsize=font_size,
#             primarycolor=Color(255, 255, 0),
#             secondarycolor=Color(0, 0, 0),
#             outlinecolor=Color(0, 0, 0),
#             backcolor=Color(0, 0, 0, 0),
#             bold=True,
#             alignment=5,
#             outline=3,
#             shadow=0,
#             marginv=margin_v 
#         )
#         subs.styles["Hormozi"] = style

#         all_words = []
#         for seg in master_index_segments:
#             if 'words' in seg:
#                 all_words.extend(seg['words'])

#         buffer = []
#         for word_data in all_words:
#             buffer.append(word_data)
#             txt = word_data['word'].strip()
#             is_end = txt[-1] in ".?!," if txt else False
            
#             if len(buffer) >= 2 or is_end:
#                 start_ms = int(buffer[0]['start'] * 1000)
#                 end_ms = int(buffer[-1]['end'] * 1000)
#                 text_content = " ".join([w['word'].strip() for w in buffer]).upper()
                
#                 subs.events.append(SSAEvent(start=start_ms, end=end_ms, text=text_content, style="Hormozi"))
#                 buffer = []
        
#         if buffer:
#             start_ms = int(buffer[0]['start'] * 1000)
#             end_ms = int(buffer[-1]['end'] * 1000)
#             text_content = " ".join([w['word'].strip() for w in buffer]).upper()
#             subs.events.append(SSAEvent(start=start_ms, end=end_ms, text=text_content, style="Hormozi"))

#         subs.save(str(output_file))

#     @staticmethod
#     def burn_subtitles(
#         video_path: str, 
#         ass_path: str, 
#         output_path: str, 
#         mode: VerticalMode = VerticalMode.CROP_CENTER,
#         facecam_coords: Tuple[int, int, int, int] = None
#     ):
#         print(f"--- Burning Subtitles (Mode: {mode.name}) ---")
        
#         total_duration = get_video_duration(video_path)
        
#         if mode == VerticalMode.CROP_CENTER:
#             vf_chain = "crop=ih*(9/16):ih,scale=1080:1920"
            
#         elif mode == VerticalMode.BLUR_BG:
#             vf_chain = (
#                 "split=2[bg][fg];"
#                 "[bg]format=yuv420p,scale=iw/4:ih/4,boxblur=10,scale=1080:1920:flags=bilinear[bg_blurred];"
#                 "[fg]scale=1080:-1[fg_scaled];"
#                 "[bg_blurred][fg_scaled]overlay=(W-w)/2:(H-h)/2"
#             )
            
#         elif mode == VerticalMode.SPLIT_SCREEN:
#             if not facecam_coords:
#                 raise ValueError("Facecam coordinates required for Split Screen.")
#             fx, fy, fw, fh = facecam_coords
            
#             vf_chain = (
#                 f"split=2[main][face];"
#                 f"[main]scale=1080:960:force_original_aspect_ratio=decrease[main_v];"
#                 f"[face]crop={fw}:{fh}:{fx}:{fy},scale=1080:960:force_original_aspect_ratio=decrease[face_v];"
#                 f"color=s=1080x1920:c=black[canvas];"
#                 f"[canvas][main_v]overlay=(W-w)/2:(960-h)/2[p1];"
#                 f"[p1][face_v]overlay=(W-w)/2:960+(960-h)/2"
#             )
#         else:
#             vf_chain = "null"

#         # Escape the path for FFmpeg filter
#         escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
        
#         if vf_chain == "null":
#             full_vf = f"ass='{escaped_ass}', setsar=1"
#         else:
#             full_vf = f"{vf_chain}[v_final];[v_final]ass='{escaped_ass}', setsar=1"

#         FFmpegWrapper.render_with_filter(
#             input_path=video_path,
#             output_path=output_path,
#             filter_complex=full_vf,
#             duration=total_duration,
#             desc=f"Burning ({mode.name})"
#         )



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

        buffer = []
        for word_data in all_words:
            buffer.append(word_data)
            txt = word_data['word'].strip()
            is_end = txt[-1] in ".?!," if txt else False
            
            if len(buffer) >= 2 or is_end:
                start_ms = int((buffer[0]['start'] - start_offset) * 1000)
                end_ms = int((buffer[-1]['end'] - start_offset) * 1000)
                text_content = " ".join([w['word'].strip() for w in buffer]).upper()
                
                subs.events.append(SSAEvent(start=start_ms, end=end_ms, text=text_content, style="Hormozi"))
                buffer = []
        
        if buffer:
            start_ms = int((buffer[0]['start'] - start_offset) * 1000)
            end_ms = int((buffer[-1]['end'] - start_offset) * 1000)
            text_content = " ".join([w['word'].strip() for w in buffer]).upper()
            subs.events.append(SSAEvent(start=start_ms, end=end_ms, text=text_content, style="Hormozi"))

        subs.save(str(output_file))

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