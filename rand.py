#Step 5: Burn subtitles
# #good hormozi transcription system
# from faster_whisper import WhisperModel
# import subprocess

# def time_fmt(seconds):
#     """ASS requires h:mm:ss.cc"""
#     h = int(seconds // 3600)
#     m = int((seconds % 3600) // 60)
#     s = int(seconds % 60)
#     cs = int((seconds % 1) * 100)
#     return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

# def generate_dynamic_ass(segments, output_file="dynamic.ass"):
#     # Header defining the "Hormozi" style
#     header = """[Script Info]
# ScriptType: v4.00+
# PlayResX: 1080
# PlayResY: 1920

# [V4+ Styles]
# Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
# # Style: Font=Impact, Size=80, Color=Yellow, Outline=Black(3px), Alignment=5(Center)
# Style: Hormozi,Impact,80,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,5,10,10,850,1

# [Events]
# Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
# """
#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write(header)
        
#         for segment in segments:
#             if not segment.words: continue
#             for word in segment.words:
#                 start_t = time_fmt(word.start)
#                 end_t = time_fmt(word.end)
#                 text = word.word.strip().upper()
                
#                 # THE MAGIC SAUCE: ASS Animation Tags
#                 # {\an5}: Align Center
#                 # {\fscx120\fscy120}: Start Scale 120%
#                 # {\t(0,80,\fscx100\fscy100)}: Transform from t=0 to t=80ms -> Scale back to 100%
#                 dynamic_text = r"{\an5\fscx130\fscy130\t(0,80,\fscx100\fscy100)}" + text
#                 print(text)
#                 f.write(f"Dialogue: 0,{start_t},{end_t},Hormozi,,0,0,0,,{dynamic_text}\n")

# def burn_dynamic(video_path):
#     # NVENC render of the animated subtitle file
#     cmd = [
#         "ffmpeg", "-y", "-i", video_path,
#         "-vf", "ass=dynamic.ass",
#         "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20",
#         "-c:a", "copy",
#         "output_hormozi_fast.mp4"
#     ]
#     subprocess.run(cmd)

# # Execution
# video_to_short = "sloppyshorts-1.mp4"
# model = WhisperModel("large-v3-turbo", device='CUDA', compute_type="float16")
# segments, _ = model.transcribe("ready_for_nvidia2.wav", language="en", word_timestamps=True, temperature=0.2, beam_size=5, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
# generate_dynamic_ass(list(segments))
# burn_dynamic(video_to_short)







### Step 1: cut out all silence in video
import torch
import subprocess
import os
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor

# --- 1. ROBUST AUDIO LOADER ---
def load_audio_robust(path, sr=16000):
    cmd = [
        "ffmpeg", "-v", "error", "-i", path, "-vn",
        "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"
    ]
    try:
        process = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFMPEG load failed: {e.stderr.decode()}")
    
    audio_np = np.frombuffer(process.stdout, np.float32).copy()
    tensor = torch.from_numpy(audio_np)
    gate_threshold = 0.015 
    mask = torch.abs(tensor) < gate_threshold
    tensor[mask] = 0.0
    return tensor

# --- 2. WORKER FUNCTION ---
def render_batch(batch_data):
    """
    Worker function to render a single batch.
    Args: (index, batch_segments, input_video, temp_dir)
    """
    i, batch, input_video, temp_dir = batch_data
    
    filter_parts = []
    concat_inputs = []
    
    for j, (start, end) in enumerate(batch):
        filter_parts.append(
            f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{j}];"
            f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{j}];"
        )
        concat_inputs.append(f"[v{j}][a{j}]")
    
    concat_filter = f"{''.join(concat_inputs)}concat=n={len(batch)}:v=1:a=1[outv][outa]"
    full_filter = "".join(filter_parts) + concat_filter
    
    filter_file = f"{temp_dir}/filter_{i}.txt"
    with open(filter_file, "w") as f:
        f.write(full_filter)
        
    chunk_name = f"{temp_dir}/chunk_{i:04d}.mp4"
    
    # NVENC SPEED COMMAND
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", input_video,
        "-filter_complex_script", filter_file,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "h264_nvenc",
        "-preset", "p2",      # <--- P2 is much faster than P7
        "-cq", "24",          # Slightly lower quality for speed (20-23 is standard)
        "-c:a", "aac", "-b:a", "192k",
        "-fps_mode", "passthrough",
        chunk_name
    ]
    
    subprocess.run(cmd)
    return chunk_name

# --- 3. MAIN CONTROLLER ---
def turbo_cut_video(
    input_video, 
    output_video, 
    padding_ms=250, 
    min_silence_ms=500,
    batch_size=40,    # Slightly smaller batches for better parallel distribution
    max_workers=3     # Consumer GPUs usually cap at 3-5 concurrent sessions
):
    print(f"--- Turbo Processing: {input_video} ---")

    # A. VAD Analysis
    print("1. Analyzing Audio...")
    wav = load_audio_robust(input_video, sr=16000)
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=False)
    (get_speech_ts, _, _, _, _) = utils
    
    speech_timestamps = get_speech_ts(
        wav, model, threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=min_silence_ms,
        return_seconds=True
    )
    
    if not speech_timestamps:
        return

    # B. Generate Segments
    prob_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_video]
    total_duration = float(subprocess.check_output(prob_cmd).strip())
    
    all_segments = []
    for ts in speech_timestamps:
        start = max(0, ts['start'] - (padding_ms / 1000))
        end = min(total_duration, ts['end'] + (padding_ms / 1000))
        
        if all_segments and start < all_segments[-1][1]:
            all_segments[-1][1] = max(all_segments[-1][1], end)
        else:
            all_segments.append([start, end])
            
    print(f"   > Found {len(all_segments)} total segments.")
    
    # C. Prepare Batches
    temp_dir = "temp_chunks_turbo"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    batches = [all_segments[i:i + batch_size] for i in range(0, len(all_segments), batch_size)]
    
    # Create arguments for workers
    # List of tuples: (index, batch, input, temp_dir)
    worker_args = [(i, b, input_video, temp_dir) for i, b in enumerate(batches)]
    
    print(f"2. Rendering {len(batches)} batches (Parallel Workers: {max_workers})...")

    # D. Execute Parallel Rendering
    chunk_files = [None] * len(batches) # Pre-allocate to ensure order
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = {executor.submit(render_batch, arg): arg[0] for arg in worker_args}
        
        # Wait for completion
        for future in futures:
        
    if not speech_timestamps:
        return

    # B. Generate Segments
    prob_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_video]
    total_duration = float(subprocess.check_output(prob_cmd).strip())
    
    all_segments = []
    for ts in speech_timestamps:
            idx = futures[future]
            try:
                chunk_name = future.result()
                chunk_files[idx] = chunk_name # Store in correct index
                print(f"   > Batch {idx+1} complete.")
            except Exception as e:
                print(f"   > Batch {idx+1} failed: {e}")

    # E. Merge
    print("3. Merging chunks...")
    concat_list_file = "turbo_merge_list.txt"
    # Ensure no None values in case of failure
    valid_chunks = [c for c in chunk_files if c]
    
    with open(concat_list_file, "w") as f:
        for chunk in valid_chunks:
            f.write(f"file '{os.path.abspath(chunk)}'\n")
            
    merge_cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_file,
        "-c", "copy",
        output_video
    ]
    subprocess.run(merge_cmd)
    
    # Cleanup
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    if os.path.exists(concat_list_file): os.remove(concat_list_file)
    print("Done!")

# Usage
turbo_cut_video("aisomniumP4.mp4", "output_turbo.mp4", max_workers=3)