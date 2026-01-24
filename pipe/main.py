from pipe.config import Config, VerticalMode
from pipe.audio_core import AudioProcessor, TimeMapper, Diarizer
from pipe.transcription import MasterIndexer
from pipe.visual_core import VisualAnalyzer
from pipe.renderer import SubtitleRenderer
# from renderer import Renderer # (Assume you move your existing burn logic here)
import torch
import gc, subprocess, os
HF_TOKEN = os.getenv("HF_ACCESS_TOKEN") 
def main(input_video: str):
    print(f"=== Processing: {input_video} ===")
    
    # 1. Audio Extraction
    raw_audio_path = Config.TEMP_DIR / "raw_audio.wav"
    # (Add extraction logic here or in AudioProcessor)
    wav_tensor = AudioProcessor.load_audio(input_video)
    
    # 2. Diarization (Who is speaking when?)
    # Returns: [(start, end, speaker_label)]
    diarization_segments = Diarizer.process(wav_tensor, HF_TOKEN)
    
    # 2. VAD & TimeMap (The Fix)
    print("Generating TimeMap...")
    keep_segments = AudioProcessor.get_vad_segments(wav_tensor)
    time_mapper = TimeMapper(keep_segments)
    
    # 3. Create Clean Audio for Whisper
    clean_audio_path = Config.TEMP_DIR / "clean_whisper.wav"
    AudioProcessor.create_clean_audio(input_video, keep_segments, clean_audio_path)
    
    # 4. Transcribe & Index
    # This index now contains RAW timestamps, perfectly synced to the original video
    master_index = MasterIndexer.create_index(clean_audio_path, time_mapper)
    MasterIndexer.inject_speakers(master_index, diarization_segments)
    MasterIndexer.save_index(master_index, Config.OUTPUT_DIR / "master_index.json")
    

    # --- PHASE 3: LOGIC & FILTERING ---
    
    recap_segments = []
    highlight_segments = []
    
    print("--- Applying Logic Filters ---")
    for i, seg in enumerate(master_index):
        text = seg['text'].lower()
        duration = seg['end'] - seg['start']
        
        # Path 1 Logic: "Recap"
        if "recap" in text:
            seg['tags'].append('recap')
            recap_segments.append(seg)
            
        # Path 2 Logic: "Highlights" (Heuristic + Visual Placeholder)
        # Only check segments > 5s and < 60s
        # if 5.0 < duration < 60.0:
        #     score = VisualAnalyzer.get_candidate_score(seg, input_video)
        #     if score >= 1: # Low threshold for demo
        #         seg['tags'].append('highlight')
        #         highlight_segments.append(seg)





    # # 5. Logic: Find Highlights (Example)
    # # We load vLLM ONCE here to manage VRAM
    # # llm = LLM(model=Config.VLLM_MODEL_ID) 
    # print(master_index)
    # print("Searching for highlights... In NEED OF HEAVY REFACTORING")
    # highlights = []
    # for segment in master_index:
    #     # Pre-filter: Only check segments > 5 seconds with "laughter" in text (simple heuristic)
    #     duration = segment['end'] - segment['start']
    #     if duration > 5 and "a" in segment['text'].lower():
            
    #         # score = VisualAnalyzer.analyze_highlight(
    #         #     input_video, segment['text'], segment['start'], segment['end'], llm_engine=llm
    #         # )
    #         score = 8 # Placeholder
            
    #         if score > 7:
    #             segment['tags'].append('highlight')
    #             highlights.append(segment)
    

    # Unload vLLM
    # del llm
    gc.collect()
    torch.cuda.empty_cache()

    # --- PHASE 4: RENDERING ---
    
    # Path 1.1: Last Recap Segment -> 9:16
    if recap_segments:
        last_recap = [recap_segments[-1]] # List of 1 dict
        print(f"Found {len(recap_segments)} recaps. Rendering last one.")
        
        # Generate ASS
        ass_path = Config.TEMP_DIR / "recap.ass"
        SubtitleRenderer.create_hormozi_ass(last_recap, str(ass_path), is_vertical=True)
        
        # Cut & Burn
        # Note: We need to trim the video to this segment first, then burn.
        # For simplicity in this script, we'll just burn the specific segment.
        # In a full prod env, you'd use the 'render_batch' logic from your original script 
        # adapted to take specific start/end times.
        
        # Simplified: Just print the action for now
        print(f"-> Ready to render Recap: {last_recap[0]['start']} - {last_recap[0]['end']}")
        
    # Path 2.1: Highlights -> 9:16 Shorts
    print(f"Found {len(highlight_segments)} highlights.")
    for i, highlight in enumerate(highlight_segments):
        print(f"-> Rendering Highlight {i+1}: {highlight['text'][:30]}...")
        
        # 1. Create ASS for this specific segment
        ass_path = Config.TEMP_DIR / f"highlight_{i}.ass"
        SubtitleRenderer.create_hormozi_ass([highlight], str(ass_path), is_vertical=True)
        
        # 2. Determine Crop Coordinates (Facecam or Active Speaker)
        # If we had facecam coords in Config:
        coords = Config.FACECAM_COORDS
        
        # 3. Render
        output_path = Config.OUTPUT_DIR / f"highlight_{i}.mp4"
        
        # We need to trim the video AND burn subs. 
        # FFmpeg complex filter: trim -> split -> (bg/fg logic) -> burn -> out
        # This is complex to construct dynamically. 
        # Strategy: Trim to temp file -> Burn.
        
        temp_trim = Config.TEMP_DIR / f"temp_trim_{i}.mp4"
        
        # Fast Seek Trim
        cmd_trim = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", str(highlight['start']),
            "-to", str(highlight['end']),
            "-i", input_video,
            "-c:v", "copy", "-c:a", "copy",
            str(temp_trim)
        ]
        subprocess.run(cmd_trim)
        
        # Burn
        SubtitleRenderer.burn_subtitles(
            str(temp_trim), 
            str(ass_path), 
            str(output_path), 
            mode=VerticalMode.BLUR_BG # Or SPLIT_SCREEN if coords exist
        )


    # 6. Render
    # Now you pass 'highlights' to your Renderer class
    # The Renderer uses segment['start'] and segment['end'] which are RAW VIDEO times.
    # It cuts, crops, and burns subs.
    # print(f"Found {len(highlights)} highlights. Ready to render.")

if __name__ == "__main__":
    main("sloppyshorts-1.mp4")