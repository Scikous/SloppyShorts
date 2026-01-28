import os
import gc
import subprocess
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional
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

class FFmpegWrapper:
    @staticmethod
    def _run_with_progress(cmd: List[str], total_duration: float, desc: str = "Processing"):
        """Internal method to run FFmpeg with a tqdm progress bar."""
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

    @staticmethod
    def extract_pcm_memory(input_path: str, sr: int = 16000) -> np.ndarray:
        """Extracts raw audio to memory as float32 numpy array."""
        cmd = ["ffmpeg", "-v", "error", "-i", str(input_path), "-vn", "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
        process = subprocess.run(cmd, capture_output=True, check=True)
        return np.frombuffer(process.stdout, np.float32).copy()

    @staticmethod
    def concat_audio_segments(input_path: str, segments: List[Tuple[float, float]], output_path: str, sr: int):
        """
        Creates a new audio file by concatenating specific segments from the input.
        Uses complex filters to avoid temp files.
        """
        filter_parts = []
        for i, (start, end) in enumerate(segments):
            filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];")
        
        concat_filter = "".join(filter_parts) + f"{''.join([f'[a{i}]' for i in range(len(segments))])}concat=n={len(segments)}:v=0:a=1[outa]"
        
        # Calculate total duration for progress bar
        total_duration = sum(end - start for start, end in segments)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-filter_complex", concat_filter,
            "-map", "[outa]",
            "-ar", str(sr),
            str(output_path)
        ]
        
        FFmpegWrapper._run_with_progress(cmd, total_duration, desc="Trimming Audio")

    @staticmethod
    def render_with_filter(
        input_path: str, 
        output_path: str, 
        filter_complex: str, 
        duration: float, 
        desc: str,
        video_codec: List[str] = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"],
        audio_codec: List[str] = ["-c:a", "copy"]
    ):
        """Generic method to render video with complex filters."""
        cmd = [
            "ffmpeg", "-y", 
            "-hwaccel", "cuda",
            "-i", str(input_path),
            "-filter_complex", filter_complex,
        ] + video_codec + audio_codec + [str(output_path)]
        
        FFmpegWrapper._run_with_progress(cmd, duration, desc=desc)

    @staticmethod
    def trim_segment(
        input_path: str,
        output_path: str,
        start: float,
        end: float,
        desc: str = "Trimming",
        copy_streams: bool = True
    ):
        """
        Simple method to trim a video segment.
        Uses stream copy by default for speed, or re-encodes if copy_streams=False.
        """
        duration = end - start
        
        if copy_streams:
            # Fast stream copy (no re-encoding)
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-ss", str(start),
                "-to", str(end),
                "-i", str(input_path),
                "-c:v", "copy",
                "-c:a", "copy",
                str(output_path)
            ]
            # Stream copy is fast, so we don't need progress bar
            subprocess.run(cmd, check=True)
        else:
            # Re-encode with progress bar
            filter_complex = f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v];[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a]"
            FFmpegWrapper.render_with_filter(
                input_path=input_path,
                output_path=output_path,
                filter_complex=filter_complex,
                duration=duration,
                desc=desc
            )

    @staticmethod
    def concat_video_segments(
        input_path: str,
        segments: List[Tuple[float, float]],
        output_path: str,
        desc: str = "Concatenating"
    ):
        """
        Concatenates multiple video segments into a single output file.
        Uses complex filters to avoid temporary files.
        
        Args:
            input_path: Source video file
            segments: List of (start, end) tuples in seconds
            output_path: Destination file
            desc: Description for progress bar
        """
        if not segments:
            raise ValueError("No segments provided for concatenation")
        
        # Build complex filter for video and audio
        filter_parts = []
        for i, (start, end) in enumerate(segments):
            # Video stream
            filter_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}];")
            # Audio stream  
            filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];")
        
        # Concatenate all segments
        video_inputs = ''.join([f'[v{i}]' for i in range(len(segments))])
        audio_inputs = ''.join([f'[a{i}]' for i in range(len(segments))])
        concat_filter = f"{video_inputs}concat=n={len(segments)}:v=1:a=0[outv];{audio_inputs}concat=n={len(segments)}:v=0:a=1[outa]"
        
        filter_complex = ''.join(filter_parts) + concat_filter
        
        # Calculate total duration
        total_duration = sum(end - start for start, end in segments)
        
        # Render with proper output mapping
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-i", str(input_path),
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path)
        ]
        
        FFmpegWrapper._run_with_progress(cmd, total_duration, desc=desc)