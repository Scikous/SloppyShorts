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