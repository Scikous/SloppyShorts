import subprocess
import os
import pysrt

def create_short_ffmpeg(video_path, srt_path, start_time, end_time, output_filename):
    duration = end_time - start_time
    
    # 1. PREPARE SUBTITLES
    # We must create a temporary subtitle file where timestamps are shifted 
    # so they align with the new cut (which starts at 00:00)
    subs = pysrt.open(srt_path)
    sliced_subs = subs.slice(starts_after=start_time*1000, ends_before=end_time*1000)
    
    # Shift timestamps back by start_time
    sliced_subs.shift(seconds=-start_time)
    
    # Save temp srt
    temp_srt = "temp_subtitles.srt"
    sliced_subs.save(temp_srt, encoding='utf-8')

    print(f"Processing {output_filename}...")

    # 2. CONSTRUCT FFMPEG COMMAND
    # We use a complex filter chain to do the "Shorts" layout
    
    # Explanation of filters:
    # [0:v]split=2[bg][fg]  --> Split input into background and foreground streams
    # [bg]scale=-2:1920...  --> Scale bg to 1920 height, crop to 1080 width, boxblur
    # [fg]scale=1080:-2     --> Scale fg to 1080 width, maintain aspect ratio
    # overlay...            --> Center fg on bg
    # subtitles...          --> Burn in the temp SRT file
    
    filter_complex = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale=-2:1920,crop=1080:1920:(iw-ow)/2:(ih-oh)/2,boxblur=20:5[bg_blurred];"
        f"[fg]scale=1080:-2[fg_scaled];"
        f"[bg_blurred][fg_scaled]overlay=(W-w)/2:(H-h)/2[labeled];"
        f"[labeled]subtitles={temp_srt}:force_style='Fontname=Arial,Fontsize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=3,Shadow=0,Alignment=2,MarginV=60'[v]"
    )

    cmd = [
        "ffmpeg",
        "-y",                   # Overwrite output
        "-ss", str(start_time), # Seek to start (fast seek)
        "-t", str(duration),    # Duration to cut
        "-i", video_path,       # Input file
        "-filter_complex", filter_complex, # The magic layout
        "-map", "[v]",          # Map the processed video
        "-map", "0:a",          # Map the audio from input
        "-c:v", "libx264",      # Video codec
        "-preset", "fast",      # Encoding speed
        "-c:a", "aac",          # Audio codec
        output_filename
    ]

    try:
        # Run FFmpeg
        subprocess.run(cmd, check=True)
        print(f"Done: {output_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    finally:
        # Cleanup temp file
        if os.path.exists(temp_srt):
            os.remove(temp_srt)

# --- CONFIGURATION ---
if __name__ == "__main__":
    
    VIDEO_FILE = "aisomniumP4.mp4"
    SRT_FILE = "test.srt"
    
    # Times in seconds (Start, End)
    clips_to_process = [
        (160, 190),  # 1:00 to 1:15
        (200, 250) # 2:00 to 2:20
    ]

    for i, (start, end) in enumerate(clips_to_process):
        create_short_ffmpeg(VIDEO_FILE, SRT_FILE, start, end, f"short_{i+1}.mp4")
