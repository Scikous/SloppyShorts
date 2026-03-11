import torch
from pathlib import Path
from enum import Enum

class VerticalMode(Enum):
    CROP_CENTER = "crop_center"
    BLUR_BG = "blur_bg"
    SPLIT_SCREEN = "split_screen"

class Config:
    # Flow Control
    CLEAN_RUN = False

    # Paths
    TEMP_DIR = Path("temp_process")
    OUTPUT_DIR = Path("output")
    
    # Models
    WHISPER_MODEL = "large-v3-turbo"
    DEMUCS_MODEL = "htdemucs_ft"
    # HuggingFace ID for vLLM (e.g., Llava-1.6 or Yi-VL)
    VLLM_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf" 
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
    
    # Settings
    SAMPLE_RATE_WHISPER = 16000
    MIN_SILENCE_MS = 500
    LMM_MAX_TOKENS = 256
    
    # Clap Detection Settings (Multi-feature spectral analysis)
    CLAP_DETECTION_ENABLED = True
    
    # Spectral thresholds
    CLAP_SPECTRAL_CENTROID_THRESHOLD = 3000.0  # Hz - higher centroid = more high-freq content (claps >3kHz, speech <2.5kHz)
    CLAP_HF_RATIO_THRESHOLD = 0.15             # High-frequency energy ratio (>4kHz), claps have broadband energy
    
    # Temporal thresholds
    CLAP_CREST_FACTOR_THRESHOLD = 5.0          # Peak/RMS ratio for transient detection (claps >5, speech 2-4)
    CLAP_MIN_DURATION_MS = 10                  # Minimum clap duration in ms
    CLAP_MAX_DURATION_MS = 150                 # Maximum clap duration in ms
    CLAP_ATTACK_TIME_MS = 8                    # Maximum attack time in ms (claps have rapid onset <8ms)
    
    # Classification threshold
    CLAP_CONFIDENCE_THRESHOLD = 0.6            # Overall weighted score threshold for clap classification
    # Note: Max possible score ~3.0 (sum of all feature weights)
    
    # Search parameters
    CLAP_MAX_SEARCH_SEC = 60                   # Search window from end of video in seconds
    CLAP_SILENCE_THRESHOLD = 0.01              # RMS below this is considered silence
    CLAP_MIN_SILENCE_MS = 200                  # Minimum gap to count as "silence" between clap and speech
    
    # Fallback Settings (if no clap detected)
    RECAP_KEYWORDS = ["recap", "review", "summary", "in this video", "let's go over", "to summarize"]
    DEFAULT_RECAP_PERCENTAGE = 0.1             # Assume last 10% is recap if no clap found

    # Create dirs
    TEMP_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

class Artifacts:
    @staticmethod
    def get_raw_audio(temp_dir: Path) -> Path:
        return temp_dir / "raw_audio.wav"
    
    @staticmethod
    def get_clean_audio(temp_dir: Path) -> Path:
        return temp_dir / "clean_whisper.wav"
    
    @staticmethod
    def get_master_index(output_dir: Path) -> Path:
        return output_dir / "master_index.json"