import os
import subprocess
import torch
import torchaudio
import gc
import numpy as np
import pysubs2
from pysubs2 import SSAEvent, SSAStyle, Color
from faster_whisper import WhisperModel
from demucs.pretrained import get_model
from demucs.apply import apply_model

# --- CONFIGURATION ---
WHISPER_MODEL_NAME = "large-v3-turbo" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

DEMUCS_MODEL_NAME = "htdemucs_ft" # High quality source separation
# DEMUCS_MODEL_NAME = "htdemucs"  # Fallback if FT crashes (lighter)

def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()

# --- MODULE 1: ROBUST AUDIO LOADING (THE FIX) ---

def load_audio_tensor(path, sr=44100):
    """
    Loads audio using FFMPEG pipes directly to a Torch Tensor.
    This bypasses torchaudio/soundfile/sox C++ conflicts.
    """
    cmd = [
        "ffmpeg",
        "-i", path,
        "-f", "f32le",        # Force 32-bit float PCM
        "-ac", "2",           # Force Stereo (Demucs expects stereo)
        "-ar", str(sr),       # Resample to Demucs native (44.1k)
        "-"                   # Pipe to stdout
    ]
    
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFMPEG failed to load audio: {e.stderr.decode()}")

    # Convert bytes to numpy -> tensor
    audio_np = np.frombuffer(out, np.float32).reshape([-1, 2]).T
    audio_tensor = torch.from_numpy(audio_np).to(DEVICE)
    return audio_tensor

def save_audio_tensor(tensor, path, sr=44100):
    """Saves the cleaned vocals to disk for Whisper to read later"""
    # Normalize to prevent clipping before saving
    tensor = tensor / (tensor.abs().max() + 1e-6)
    
    # Convert to CPU numpy
    audio_np = tensor.cpu().numpy().T
    
    import soundfile as sf
    sf.write(path, audio_np, sr)


#Step 2: Extract only vocals from video -- Only necessary for audio with background noise
# --- MODULE 2: NATIVE DEMUCS INFERENCE ---

def extract_vocals_native(input_path, output_path):
    print(f"--- [Tier 2] Separating Vocals: {os.path.basename(input_path)} ---")
    
    # 1. Load Model
    # We load the model directly via API
    model = get_model(DEMUCS_MODEL_NAME)
    model.to(DEVICE)
    
    # 2. Load Audio (Safe Method)
    wav = load_audio_tensor(input_path, sr=44100)
    
    # 3. Inference
    # 'shifts=0' disables the "bag" averaging if it's unstable. 
    # Set shifts=1 for slightly better quality if stable.
    print("   > Running Inference (This may take time)...")
    
    # Using 'split=True' handles memory management internally by chunking
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    
    with torch.no_grad():
        sources = apply_model(
            model, 
            wav[None], 
            device=DEVICE, 
            shifts=0,   # Set to 0 to prevent "bag" crashes
            split=True, # Critical for long files
            overlap=0.25, 
            progress=True
        )
    
    # sources shape: [Batch, Sources, Channels, Time]
    # htdemucs sources: ["drums", "bass", "other", "vocals"]
    # We want index 3 (vocals)
    vocals = sources[0, 3] * ref.std() + ref.mean()
    
    print("   > Saving Vocals...")
    save_audio_tensor(vocals, output_path, sr=44100)
    
    # Cleanup
    del model, sources, wav
    cleanup_gpu()
    return output_path

#Step 3: Transcribe -- Needs to decouple the .ASS file creation
#Step 4: Create .ASS hormozi (or other styled) subtitles
#Step 5: Place and burn subtitles into video
# --- MODULE 3: WHISPER & STYLING ---

def generate_hormozi_subs(audio_path, output_ass):
    print(f"--- [Tier 1] Transcribing & Styling ---")
    
    # Load Whisper
    model = WhisperModel("large-v3-turbo", device=DEVICE, compute_type=COMPUTE_TYPE)
    
    # Transcribe with internal VAD
    segments, _ = model.transcribe(
        audio_path,
        beam_size=10,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True
    )
    
    # Initialize .ASS
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = "1080"
    subs.info["PlayResY"] = "1920"
    
    # Define Hormozi Style
    hormozi = SSAStyle(
        # name="Hormozi",
        fontname="Impact",
        fontsize=80,
        primarycolor=Color(255, 255, 0),    # Yellow
        secondarycolor=Color(0, 0, 0),      # Black
        outlinecolor=Color(0, 0, 0),
        backcolor=Color(0, 0, 0, 0),
        bold=True,
        alignment=5, # Center
        outline=3,
        shadow=0,
        marginv=850  # Position ~1/3 from bottom
    )
    subs.styles["Hormozi"] = hormozi

    # Process Words
    buffer_words = []
    
    all_words = []
    for s in segments:
        if s.words: all_words.extend(s.words)
            
    print("   > Generating dynamic captions...")
    
    for word in all_words:
        buffer_words.append(word)
        
        # Logic: Break on punctuation OR every 2 words
        # This keeps the pace fast
        is_end_of_sentence = word.word.strip()[-1] in ".?!,"
        if len(buffer_words) >= 2 or is_end_of_sentence:
            
            start_ms = int(buffer_words[0].start * 1000)
            end_ms = int(buffer_words[-1].end * 1000)
            
            # Text transformation: UPPERCASE
            text_content = " ".join([w.word.strip() for w in buffer_words]).upper()
            
            event = SSAEvent(
                start=start_ms,
                end=end_ms,
                text=text_content,
                style="Hormozi"
            )
            subs.events.append(event)
            buffer_words = []
            
    subs.save(output_ass)
    print(f"--- Done. Saved to {output_ass} ---")



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

# --- MODULE 2: CONTEXT-AWARE TRANSCRIPTION ---
def transcribe_with_context(audio_tensor, timestamps):
    print(f"--- [Tier 1] Transcribing with {WHISPER_MODEL_NAME} ---")
    
    model = WhisperModel(WHISPER_MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    
    full_words = []
    previous_context = "" 
    total = len(timestamps)
    
    # Ensure audio is on CPU/Numpy for Whisper
    audio_np = audio_tensor.numpy()
    
    for i, segment in enumerate(timestamps):
        start_sec = segment['start']
        end_sec = segment['end']
        
        # Calculate indices
        start_sample = int(start_sec * 16000)
        end_sample = int(end_sec * 16000)
        
        # Pad slightly to catch word boundaries
        pad = 1600 # 0.1s
        s = max(0, start_sample - pad)
        e = min(len(audio_np), end_sample + pad)
        
        audio_slice = audio_np[s:e]
        
        # --- CRITICAL FIXES HERE ---
        segments_gen, _ = model.transcribe(
            audio_slice,
            beam_size=5,
            language="en",          # <--- FORCE ENGLISH (Stops Kanji/Spanish)
            temperature=0.0,        # <--- GREEDY DECODING (Stops creativity)
            initial_prompt=previous_context, 
            word_timestamps=True,
            condition_on_previous_text=False, # <--- DISCONNECT CONTEXT IF HALLUCINATING
            no_speech_threshold=0.6 # <--- If whisper thinks it's silence, believe it
        )
        
        current_text = ""
        
        for s_gen in segments_gen:
            # Secondary Hallucination Filter:
            # If the segment has high "no_speech_prob" (returned by Whisper), ignore it.
            if s_gen.no_speech_prob > 0.6:
                continue

            current_text += s_gen.text + " "
            if s_gen.words:
                for word in s_gen.words:
                    slice_start_sec = s / 16000
                    
                    # Filter empty strings AGAIN
                    clean_word = word.word.strip()
                    
                    # Filter common hallucination triggers
                    # Whisper often outputs just "." or "..." or "?" on silence
                    if clean_word and clean_word not in [".", "...", "?", "!", ","]:
                        full_words.append({
                            "start": word.start + slice_start_sec,
                            "end": word.end + slice_start_sec,
                            "word": clean_word
                        })
        
        # Only update context if we actually got text
        if len(current_text) > 5: 
            previous_context = current_text[-200:]
        
        if i % 10 == 0:
            print(f"   > Processed segment {i}/{total}")

    cleanup_gpu()
    return full_words

# --- MODULE 3: STYLING (Fixed) ---
def generate_ass(word_list, output_file):
    print("--- [Tier 1] Generating .ASS ---")
    
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = "1080"
    subs.info["PlayResY"] = "1920"
    
    # Define Style
    hormozi = SSAStyle(
        fontname="Impact",
        fontsize=80,
        primarycolor=Color(255, 255, 0),    # Yellow
        secondarycolor=Color(0, 0, 0),      # Black
        outlinecolor=Color(0, 0, 0),
        backcolor=Color(0, 0, 0, 0),
        bold=True,
        alignment=5, 
        outline=3,
        shadow=0,
        marginv=850
    )
    subs.styles["Hormozi"] = hormozi

    buffer_words = []
    
    for word_obj in word_list:
        buffer_words.append(word_obj)
        
        # --- THE FIX IS HERE ---
        # Check if word exists and has length > 0 before checking [-1]
        word_text = word_obj["word"]
        is_end = (len(word_text) > 0 and word_text[-1] in ".?!,")
        # -----------------------

        if len(buffer_words) >= 2 or is_end:
            
            start_ms = int(buffer_words[0]["start"] * 1000)
            end_ms = int(buffer_words[-1]["end"] * 1000)
            
            text = " ".join([w["word"] for w in buffer_words]).upper()
            
            event = SSAEvent(
                start=start_ms,
                end=end_ms,
                text=text,
                style="Hormozi"
            )
            subs.events.append(event)
            buffer_words = []
            
    subs.save(output_file)
    print(f"--- Saved to {output_file} ---")
# --- MAIN ---

if __name__ == "__main__":
    VIDEO_FILE = "short_1.mp4" # Your Input
    VOCALS_FILE = "temp_vocals_only.wav"
    OUTPUT_ASS = "output_hormozi.ass"
    
    try:
        # Step 1: Extract Vocals (Native Python)
        if not os.path.exists(VOCALS_FILE):
            extract_vocals_native(VIDEO_FILE, VOCALS_FILE)
        else:
            print("Found existing vocals file, skipping separation.")
            
        # audio_16k, timestamps = get_vad_timestamps(VOCALS_FILE)
        
        # # 2. Transcribe Loop
        # word_data = transcribe_with_context(audio_16k, timestamps)
        
        # # # 3. Create Subtitles
        # print(word_data)
        # generate_ass(word_data, OUTPUT_ASS)
        # Step 2: Transcribe
        generate_hormozi_subs(VOCALS_FILE, OUTPUT_ASS)
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        # Detailed traceback usually helps here
        import traceback
        traceback.print_exc()
