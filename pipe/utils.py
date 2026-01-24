import os
import gc
import subprocess
import torch
from tqdm import tqdm
from typing import List
from pipe.config import Config

def cleanup_gpu():
    """Forces VRAM cleanup."""
    if Config.DEVICE == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

def get_video_duration(video_path: str) -> float:
    """Returns video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    return float(subprocess.check_output(cmd).strip())

def run_ffmpeg_with_progress(cmd: List[str], total_duration: float, desc: str = "Processing"):
    """Runs FFmpeg with a live tqdm progress bar."""
    # Inject progress monitoring flags
    cmd_with_progress = cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:]
    
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    process = subprocess.Popen(
        cmd_with_progress, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True,
        startupinfo=startupinfo
    )
    
    with tqdm(total=total_duration, desc=desc, unit="s", bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s") as pbar:
        for line in process.stdout:
            if "out_time_us=" in line:
                try:
                    us = int(line.split("=")[1].strip())
                    sec = us / 1_000_000
                    pbar.n = min(sec, total_duration)
                    pbar.refresh()
                except ValueError:
                    pass
                    
    process.wait()
    if process.returncode != 0:
        err = process.stderr.read()
        raise RuntimeError(f"FFmpeg Error: {err}")