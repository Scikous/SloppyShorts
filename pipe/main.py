# from pipe.config import Config, VerticalMode
# from pipe.audio_core import AudioProcessor, TimeMapper, Diarizer
# from pipe.transcription import MasterIndexer, SubtitleGenerator
# from pipe.visual_core import VisualAnalyzer
# from pipe.renderer import SubtitleRenderer
# from pipe.utils import cleanup_gpu, get_video_duration, FFmpegWrapper
# import gc
# import json
# import os
# from pathlib import Path

# HF_TOKEN = os.getenv("HF_ACCESS_TOKEN")

# def main(input_video: str):
#     print(f"=== Processing: {input_video} ===")
    
#     # --- Define Paths for Caching ---
#     raw_audio_path = Config.TEMP_DIR / "raw_audio.wav"
#     clean_audio_path = Config.TEMP_DIR / "clean_whisper.wav"
#     vad_segments_path = Config.TEMP_DIR / "vad_segments.json"
#     diarization_path = Config.TEMP_DIR / "diarization.json"
#     master_index_path = Config.OUTPUT_DIR / "master_index.json"
    
#     # --- PHASE 1: AUDIO & TRANSCRIPTION (With Resume Logic) ---
    
#     # 1. Check for Master Index (The "Done" State)
#     if master_index_path.exists() and vad_segments_path.exists():
#         print("-> Found existing Master Index & VAD data. Loading from cache...")
#         master_index = MasterIndexer.load_index(master_index_path)
#         keep_segments = AudioProcessor.load_segments(vad_segments_path)
#         time_mapper = TimeMapper(keep_segments)
#     else:
#         # 2. Raw Audio Extraction
#         if raw_audio_path.exists():
#             print("-> Found cached raw audio.")
#         else:
#             print("-> Extracting raw audio...")
#             AudioProcessor.extract_raw_audio(input_video, raw_audio_path)
        
#         # 3. VAD (Voice Activity Detection)
#         if vad_segments_path.exists():
#             print("-> Found cached VAD segments.")
#             keep_segments = AudioProcessor.load_segments(vad_segments_path)
#         else:
#             print("-> Running VAD...")
#             wav_tensor = AudioProcessor.load_audio(raw_audio_path)
#             keep_segments = AudioProcessor.get_vad_segments(wav_tensor)
#             AudioProcessor.save_segments(keep_segments, vad_segments_path)
#             del wav_tensor
#             gc.collect()
            
#         time_mapper = TimeMapper(keep_segments)
        
#         # 4. Clean Audio for Whisper
#         if clean_audio_path.exists():
#             print("-> Found cached clean audio.")
#         else:
#             print("-> Creating clean audio...")
#             # Use FFmpegWrapper for audio segment concatenation
#             AudioProcessor.create_clean_audio(str(raw_audio_path), keep_segments, clean_audio_path)
            
#         # 5. Diarization
#         diarization_segments = []
#         if diarization_path.exists():
#             print("-> Found cached Diarization.")
#             with open(diarization_path, 'r') as f:
#                 diarization_segments = json.load(f)
#         else:
#             print("-> Running Diarization...")
#             # Diarizer now accepts path to save memory
#             diarization_segments = Diarizer.process(str(raw_audio_path), HF_TOKEN)
#             with open(diarization_path, 'w') as f:
#                 json.dump(diarization_segments, f)
        
#         # 6. Transcribe & Index
#         print("-> Transcribing & Indexing...")
#         master_index = MasterIndexer.create_index(clean_audio_path, time_mapper)
#         MasterIndexer.inject_speakers(master_index, diarization_segments)
#         MasterIndexer.save_index(master_index, master_index_path)

#     # --- PHASE 2: LOGIC & FILTERING ---
    
#     recap_segments = []
#     highlight_segments = []
    
#     print("--- Applying Logic Filters ---")
#     for seg in master_index:
#         text = seg['text'].lower()
#         duration = seg['end'] - seg['start']
        
#         # Path 1 Logic: "Recap"
#         if "recap" in text:
#             seg['tags'].append('recap')
#             recap_segments.append(seg)
            
#         # Path 2 Logic: "Highlights"
#         # TODO: Future Enhancement - Replace placeholder with VisualAnalyzer.get_candidate_score()
#         # This is where the "Highlight Finder" logic will be expanded:
#         # - Extract keyframes from video segments
#         # - Analyze visual features (action detection, scene changes, etc.)
#         # - Score segments based on visual interest + audio energy
#         # - Apply ML-based ranking for true highlight detection
#         if 5.0 < duration < 60.0:
#             score = 1  # Placeholder for VisualAnalyzer.get_candidate_score(seg, input_video)
#             if score >= 1:
#                 seg['tags'].append('highlight')
#                 highlight_segments.append(seg)

#     # Aggressive Memory Cleanup After Heavy Indexing Phase
#     print("-> Cleaning up memory before rendering phase...")
#     cleanup_gpu()
#     gc.collect()

#     # --- PHASE 3: RENDERING (With Resume Logic) ---
    
#     # Path 1.1: Recap -> 9:16 Vertical Format
#     if recap_segments:
#         last_recap = recap_segments[-1]
#         output_recap = Config.OUTPUT_DIR / f"{Path(input_video).stem}_recap_9_16.mp4"
        
#         if output_recap.exists():
#             print(f"-> Skipping Recap (Exists): {output_recap}")
#         else:
#             print(f"\n--- Path 1.1: Rendering Recap ({last_recap['start']:.2f}s - {last_recap['end']:.2f}s) ---")
            
#             # 1. Generate ASS subtitles
#             ass_path = Config.TEMP_DIR / "recap.ass"
#             SubtitleRenderer.create_hormozi_ass([last_recap], str(ass_path), is_vertical=True)
            
#             # 2. Extract segment using FFmpegWrapper
#             temp_trim = Config.TEMP_DIR / "temp_recap_raw.mp4"
#             FFmpegWrapper.trim_segment(
#                 input_path=input_video,
#                 output_path=str(temp_trim),
#                 start=last_recap['start'],
#                 end=last_recap['end'],
#                 desc="Extracting Recap",
#                 copy_streams=True  # Fast stream copy
#             )
            
#             # 3. Burn subtitles and verticalize
#             SubtitleRenderer.burn_subtitles(
#                 str(temp_trim), 
#                 str(ass_path), 
#                 str(output_recap), 
#                 mode=VerticalMode.BLUR_BG  # Change to SPLIT_SCREEN if Config.FACECAM_COORDS is set
#             )
#             print(f"-> Recap saved to {output_recap}")

#     # Path 1.2: Clean Long-form (No Silence, No Recaps)
#     output_clean = Config.OUTPUT_DIR / f"{Path(input_video).stem}_clean_16_9.mp4"
#     srt_path = output_clean.with_suffix(".srt")
    
#     if output_clean.exists() and srt_path.exists():
#         print(f"-> Skipping Clean Video & SRT (Exists): {output_clean}")
#     else:
#         print(f"\n--- Path 1.2: Rendering Clean Long-form ---")
        
#         # 1. Calculate intervals to keep (VAD segments - Recap segments)
#         recap_ranges = [(r['start'], r['end']) for r in recap_segments]
#         final_intervals = time_mapper.get_clean_intervals(keep_segments, recap_ranges)
        
#         # 2. Use FFmpegWrapper to concatenate video segments directly
#         FFmpegWrapper.concat_video_segments(
#             input_path=input_video,
#             segments=final_intervals,
#             output_path=str(output_clean),
#             desc="Creating Clean Video"
#         )
#         print(f"-> Clean video saved to {output_clean}")
        
#         # 3. Generate SRT subtitles for the clean video
#         print("-> Generating SRT subtitles...")
#         SubtitleGenerator.generate_srt(master_index, final_intervals, str(srt_path))
#         print(f"-> SRT subtitles saved to {srt_path}")

#     # Path 2.1: Highlights -> 9:16 Shorts
#     print(f"\n--- Processing {len(highlight_segments)} Highlights ---")
#     for i, highlight in enumerate(highlight_segments):
#         output_path = Config.OUTPUT_DIR / f"highlight_{i}.mp4"
#         if output_path.exists():
#             print(f"-> Skipping Highlight {i} (Exists)")
#             continue
            
#         print(f"-> Rendering Highlight {i+1}: {highlight['text'][:50]}...")
        
#         # 1. Generate ASS subtitles
#         ass_path = Config.TEMP_DIR / f"highlight_{i}.ass"
#         SubtitleRenderer.create_hormozi_ass([highlight], str(ass_path), is_vertical=True)
        
#         # 2. Extract highlight segment using FFmpegWrapper
#         temp_trim = Config.TEMP_DIR / f"temp_trim_{i}.mp4"
#         FFmpegWrapper.trim_segment(
#             input_path=input_video,
#             output_path=str(temp_trim),
#             start=highlight['start'],
#             end=highlight['end'],
#             desc=f"Extracting Highlight {i+1}",
#             copy_streams=True  # Fast stream copy
#         )
        
#         # 3. Burn subtitles and verticalize
#         SubtitleRenderer.burn_subtitles(
#             str(temp_trim), 
#             str(ass_path), 
#             str(output_path), 
#             mode=VerticalMode.BLUR_BG  # Or SPLIT_SCREEN if Config.FACECAM_COORDS is set
#         )
        
#         print(f"-> Highlight {i+1} saved to {output_path}")
        
#         # Clean up temp file
#         if temp_trim.exists():
#             temp_trim.unlink()

#     print("\n=== Processing Complete ===")
#     print(f"Outputs saved to: {Config.OUTPUT_DIR}")
    
#     # Final cleanup
#     cleanup_gpu()

# if __name__ == "__main__":
#     main("sloppyshorts-1.mp4")





from pipe.config import Config, VerticalMode
from pipe.audio_core import AudioProcessor, TimeMapper, Diarizer
from pipe.transcription import MasterIndexer, SubtitleGenerator
from pipe.visual_core import VisualAnalyzer
from pipe.renderer import SubtitleRenderer
from pipe.utils import cleanup_gpu, get_video_duration, FFmpegWrapper
import gc
import json
import os
from pathlib import Path

HF_TOKEN = os.getenv("HF_ACCESS_TOKEN")

def main(input_video: str):
    print(f"=== Processing: {input_video} ===")
    
    # --- Define Paths for Caching ---
    raw_audio_path = Config.TEMP_DIR / "raw_audio.wav"
    clean_audio_path = Config.TEMP_DIR / "clean_whisper.wav"
    vad_segments_path = Config.TEMP_DIR / "vad_segments.json"
    diarization_path = Config.TEMP_DIR / "diarization.json"
    master_index_path = Config.OUTPUT_DIR / "master_index.json"
    
    # --- PHASE 1: AUDIO & TRANSCRIPTION (With Resume Logic) ---
    
    # 1. Check for Master Index (The "Done" State)
    if master_index_path.exists() and vad_segments_path.exists():
        print("-> Found existing Master Index & VAD data. Loading from cache...")
        master_index = MasterIndexer.load_index(master_index_path)
        keep_segments = AudioProcessor.load_segments(vad_segments_path)
        time_mapper = TimeMapper(keep_segments)
    else:
        # 2. Raw Audio Extraction
        if raw_audio_path.exists():
            print("-> Found cached raw audio.")
        else:
            print("-> Extracting raw audio...")
            AudioProcessor.extract_raw_audio(input_video, raw_audio_path)
        
        # 3. VAD (Voice Activity Detection)
        if vad_segments_path.exists():
            print("-> Found cached VAD segments.")
            keep_segments = AudioProcessor.load_segments(vad_segments_path)
        else:
            print("-> Running VAD...")
            wav_tensor = AudioProcessor.load_audio(raw_audio_path)
            keep_segments = AudioProcessor.get_vad_segments(wav_tensor)
            AudioProcessor.save_segments(keep_segments, vad_segments_path)
            del wav_tensor
            gc.collect()
            
        time_mapper = TimeMapper(keep_segments)
        
        # 4. Clean Audio for Whisper
        if clean_audio_path.exists():
            print("-> Found cached clean audio.")
        else:
            print("-> Creating clean audio...")
            # Use FFmpegWrapper for audio segment concatenation
            AudioProcessor.create_clean_audio(str(raw_audio_path), keep_segments, clean_audio_path)
            
        # 5. Diarization
        diarization_segments = []
        if diarization_path.exists():
            print("-> Found cached Diarization.")
            with open(diarization_path, 'r') as f:
                diarization_segments = json.load(f)
        else:
            print("-> Running Diarization...")
            # Diarizer now accepts path to save memory
            diarization_segments = Diarizer.process(str(raw_audio_path), HF_TOKEN)
            with open(diarization_path, 'w') as f:
                json.dump(diarization_segments, f)
        
        # 6. Transcribe & Index
        print("-> Transcribing & Indexing...")
        master_index = MasterIndexer.create_index(clean_audio_path, time_mapper)
        MasterIndexer.inject_speakers(master_index, diarization_segments)
        MasterIndexer.save_index(master_index, master_index_path)

    # --- PHASE 2: LOGIC & FILTERING ---
    
    recap_segments = []
    highlight_segments = []
    
    print("--- Applying Logic Filters ---")
    for seg in master_index:
        text = seg['text'].lower()
        duration = seg['end'] - seg['start']
        
        # Path 1 Logic: "Recap"
        if "recap" in text:
            seg['tags'].append('recap')
            recap_segments.append(seg)
            
        # Path 2 Logic: "Highlights"
        # TODO: Future Enhancement - Replace placeholder with VisualAnalyzer.get_candidate_score()
        # This is where the "Highlight Finder" logic will be expanded:
        # - Extract keyframes from video segments
        # - Analyze visual features (action detection, scene changes, etc.)
        # - Score segments based on visual interest + audio energy
        # - Apply ML-based ranking for true highlight detection
        if 5.0 < duration < 60.0:
            score = 1  # Placeholder for VisualAnalyzer.get_candidate_score(seg, input_video)
            if score >= 1:
                seg['tags'].append('highlight')
                highlight_segments.append(seg)

    # Aggressive Memory Cleanup After Heavy Indexing Phase
    print("-> Cleaning up memory before rendering phase...")
    cleanup_gpu()
    gc.collect()

    # --- PHASE 3: RENDERING (With Resume Logic) ---
    
    # Path 1.1: Recap -> 9:16 Vertical Format
    if recap_segments:
        last_recap = recap_segments[-1]
        output_recap = Config.OUTPUT_DIR / f"{Path(input_video).stem}_recap_9_16.mp4"
        
        if output_recap.exists():
            print(f"-> Skipping Recap (Exists): {output_recap}")
        else:
            print(f"\n--- Path 1.1: Rendering Recap ({last_recap['start']:.2f}s - end of video) ---")
            
            # 1. Find all segments from last_recap to end of video
            last_recap_idx = master_index.index(last_recap)
            recap_sequence = master_index[last_recap_idx:]
            
            # 2. Get total video duration
            total_video_duration = get_video_duration(input_video)
            
            # 3. Generate ASS subtitles with start_offset for proper alignment
            ass_path = Config.TEMP_DIR / "recap.ass"
            SubtitleRenderer.create_hormozi_ass(
                recap_sequence, 
                str(ass_path), 
                is_vertical=True,
                start_offset=last_recap['start']
            )
            
            # 4. Extract segment from last_recap start to end of video
            temp_trim = Config.TEMP_DIR / "temp_recap_raw.mp4"
            FFmpegWrapper.trim_segment(
                input_path=input_video,
                output_path=str(temp_trim),
                start=last_recap['start'],
                end=total_video_duration,
                desc="Extracting Recap",
                copy_streams=True  # Fast stream copy
            )
            
            # 5. Burn subtitles and verticalize
            SubtitleRenderer.burn_subtitles(
                str(temp_trim), 
                str(ass_path), 
                str(output_recap), 
                mode=VerticalMode.BLUR_BG  # Change to SPLIT_SCREEN if Config.FACECAM_COORDS is set
            )
            print(f"-> Recap saved to {output_recap}")

    # Path 1.2: Clean Long-form (No Silence, No Recaps)
    output_clean = Config.OUTPUT_DIR / f"{Path(input_video).stem}_clean_16_9.mp4"
    srt_path = output_clean.with_suffix(".srt")
    
    if output_clean.exists() and srt_path.exists():
        print(f"-> Skipping Clean Video & SRT (Exists): {output_clean}")
    else:
        print(f"\n--- Path 1.2: Rendering Clean Long-form ---")
        
        # 1. Calculate intervals to keep (VAD segments - Recap segments)
        recap_ranges = [(r['start'], r['end']) for r in recap_segments]
        final_intervals = time_mapper.get_clean_intervals(keep_segments, recap_ranges)
        
        # 2. Use FFmpegWrapper to concatenate video segments directly
        FFmpegWrapper.concat_video_segments(
            input_path=input_video,
            segments=final_intervals,
            output_path=str(output_clean),
            desc="Creating Clean Video"
        )
        print(f"-> Clean video saved to {output_clean}")
        
        # 3. Generate SRT subtitles for the clean video
        print("-> Generating SRT subtitles...")
        SubtitleGenerator.generate_srt(master_index, final_intervals, str(srt_path))
        print(f"-> SRT subtitles saved to {srt_path}")

    # Path 2.1: Highlights -> 9:16 Shorts
    print(f"\n--- Processing {len(highlight_segments)} Highlights ---")
    for i, highlight in enumerate(highlight_segments):
        output_path = Config.OUTPUT_DIR / f"highlight_{i}.mp4"
        if output_path.exists():
            print(f"-> Skipping Highlight {i} (Exists)")
            continue
            
        print(f"-> Rendering Highlight {i+1}: {highlight['text'][:50]}...")
        
        # 1. Generate ASS subtitles with start_offset for proper alignment
        ass_path = Config.TEMP_DIR / f"highlight_{i}.ass"
        SubtitleRenderer.create_hormozi_ass(
            [highlight], 
            str(ass_path), 
            is_vertical=True,
            start_offset=highlight['start']
        )
        
        # 2. Extract highlight segment using FFmpegWrapper
        temp_trim = Config.TEMP_DIR / f"temp_trim_{i}.mp4"
        FFmpegWrapper.trim_segment(
            input_path=input_video,
            output_path=str(temp_trim),
            start=highlight['start'],
            end=highlight['end'],
            desc=f"Extracting Highlight {i+1}",
            copy_streams=True  # Fast stream copy
        )
        
        # 3. Burn subtitles and verticalize
        SubtitleRenderer.burn_subtitles(
            str(temp_trim), 
            str(ass_path), 
            str(output_path), 
            mode=VerticalMode.BLUR_BG  # Or SPLIT_SCREEN if Config.FACECAM_COORDS is set
        )
        
        print(f"-> Highlight {i+1} saved to {output_path}")
        
        # Clean up temp file
        if temp_trim.exists():
            temp_trim.unlink()

    print("\n=== Processing Complete ===")
    print(f"Outputs saved to: {Config.OUTPUT_DIR}")
    
    # Final cleanup
    cleanup_gpu()

if __name__ == "__main__":
    main("sloppyshorts-1.mp4")