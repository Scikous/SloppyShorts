import argparse
import os
import gc
from pathlib import Path

from pipe.config import Config
from pipe.utils import cleanup_gpu
from pipe.audio_core import AudioProcessor, TimeMapper
from pipe.vad import VADProcessor
from pipe.diarization import Diarizer
from pipe.transcription import Transcriber
from pipe.indexing import MasterIndexer
from pipe.rendering import VideoRenderer

HF_TOKEN = os.getenv("HF_ACCESS_TOKEN")

def process_video(
    input_video: str, 
    run_vad=True, 
    run_diarization=True, 
    run_transcription=True, 
    run_indexing=True, 
    run_rendering=True
):
    print(f"=== Processing Pipeline: {input_video} ===")
    
    # --- Cache Paths ---
    raw_audio_path = Config.TEMP_DIR / "raw_audio.wav"
    clean_audio_path = Config.TEMP_DIR / "clean_whisper.wav"
    vad_segments_path = Config.TEMP_DIR / "vad_segments.json"
    diarization_path = Config.TEMP_DIR / "diarization.json"
    transcription_path = Config.TEMP_DIR / "transcription.json"
    master_index_path = Config.OUTPUT_DIR / "master_index.json"

    # --- Phase 1: Audio Extraction ---
    if not raw_audio_path.exists():
        print("-> Extracting raw audio...")
        AudioProcessor.extract_raw_audio(input_video, str(raw_audio_path))
    else:
        print("-> Found cached raw audio.")

    # --- Phase 2: VAD ---
    keep_segments =[]
    if run_vad:
        keep_segments = VADProcessor.run(str(raw_audio_path), str(vad_segments_path))
    else:
        keep_segments = VADProcessor.load_segments(str(vad_segments_path))
        print("-> Loaded cached VAD segments.")
        
    time_mapper = TimeMapper(keep_segments)

    # --- Phase 3: Clean Audio ---
    if not clean_audio_path.exists():
        print("-> Creating clean audio...")
        AudioProcessor.create_clean_audio(str(raw_audio_path), keep_segments, str(clean_audio_path))
    else:
        print("-> Found cached clean audio.")

    # --- Phase 4: Diarization (Now uses Clean Audio) ---
    diarization_data =[]
    if run_diarization:
        # TimeMapper converts the 'clean audio timestamps' back to 'raw video timestamps' on the fly.
        diarization_data = Diarizer.run(str(clean_audio_path), str(diarization_path), HF_TOKEN, time_mapper)
    else:
        if diarization_path.exists():
            import json
            with open(diarization_path, 'r') as f:
                diarization_data =[tuple(x) for x in json.load(f)]
            print("-> Loaded cached Diarization.")

    # --- Phase 5: Transcription ---
    transcription_data =[]
    if run_transcription:
        transcription_data = Transcriber.run(str(clean_audio_path), str(transcription_path))
    else:
        if transcription_path.exists():
            import json
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)
            print("-> Loaded cached Transcription.")

    # --- Phase 6: Indexing ---
    master_index =[]
    if run_indexing:
        if transcription_data and diarization_data:
            master_index = MasterIndexer.run(transcription_data, diarization_data, time_mapper, str(master_index_path))
        else:
            print("-> Missing transcription or diarization data to build index. Run those phases first.")
    else:
        if master_index_path.exists():
            import json
            with open(master_index_path, 'r', encoding='utf-8') as f:
                master_index = json.load(f)
            print("-> Loaded cached Master Index.")

    # Aggressive memory cleanup after the heavy ML processing phases
    cleanup_gpu()
    gc.collect()

    # --- Phase 7: Rendering ---
    if run_rendering:
        if master_index and keep_segments:
            VideoRenderer.run(input_video, master_index, time_mapper, keep_segments)
        else:
            print("-> Missing master index or VAD segments required for rendering.")

    print("\n=== Processing Complete ===")
    print(f"Outputs saved to: {Config.OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Modular Video Processing Pipeline")
    parser.add_argument("input_video", type=str, nargs='?', default="sloppyshorts-1.mp4", help="Path to input video")
    parser.add_argument("--skip-vad", action="store_true", help="Skip VAD and use cache")
    parser.add_argument("--skip-diarization", action="store_true", help="Skip Diarization and use cache")
    parser.add_argument("--skip-transcription", action="store_true", help="Skip Transcription and use cache")
    parser.add_argument("--skip-indexing", action="store_true", help="Skip Indexing and use cache")
    parser.add_argument("--skip-rendering", action="store_true", help="Skip Video Rendering")
    
    args = parser.parse_args()
    print(args.skip_vad)
    process_video(
        input_video=args.input_video, 
        run_vad=not args.skip_vad,
        run_diarization=not args.skip_diarization,
        run_transcription=not args.skip_transcription,
        run_indexing=not args.skip_indexing,
        run_rendering=not args.skip_rendering
    )

if __name__ == "__main__":
    main()