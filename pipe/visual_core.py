import torch
# from vllm import LLM, SamplingParams
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from PIL import Image
from pipe.config import Config
import cv2
from typing import List

class VisualAnalyzer:
    @staticmethod
    def detect_scenes(video_path: str, start_time: float, end_time: float) -> List[float]:
        """Returns a list of timestamps (seconds) representing keyframes within the segment."""
        video_manager = VideoManager([str(video_path)])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30.0))
        
        # Set duration
        video_manager.set_duration(start_time=start_time, end_time=end_time)
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        
        scene_list = scene_manager.get_scene_list()
        # Get the middle frame of each scene
        timestamps = []
        for scene in scene_list:
            mid_frame = (scene[0].get_seconds() + scene[1].get_seconds()) / 2
            timestamps.append(mid_frame)
            
        if not timestamps: # Fallback if no scene change detected
            timestamps.append((start_time + end_time) / 2)
            
        video_manager.release()
        return timestamps

    @staticmethod
    def extract_frame(video_path: str, timestamp: float) -> Image.Image:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = cap.read()
        cap.release()
        if success:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return None

    @staticmethod
    def analyze_highlight(video_path: str, transcript_text: str, start: float, end: float, llm_engine=None):
        """
        Uses vLLM to score a segment. 
        Note: llm_engine should be passed in to avoid reloading model per call.
        """
        keyframes_ts = VisualAnalyzer.detect_scenes(video_path, start, end)
        
        # Limit to max 3 frames to save tokens
        keyframes_ts = keyframes_ts[:3] 
        
        images = []
        for ts in keyframes_ts:
            img = VisualAnalyzer.extract_frame(video_path, ts)
            if img: images.append(img)
            
        if not images: return 0

        # Construct Prompt (Simplified for vLLM LLaVA)
        # Note: Actual vLLM image passing syntax depends on version (Multi-modal support)
        # This is pseudo-code for the logic flow
        prompt = f"""
        USER: <image>\nAnalyze this video segment.
        Transcript: "{transcript_text}"
        Is this moment funny, viral, or a highlight? 
        Reply with JSON: {{"score": 1-10, "reason": "..."}}
        ASSISTANT:
        """
        
        # In a real implementation, you would construct the vLLM inputs object here
        # inputs = ...
        # output = llm_engine.generate(inputs, sampling_params)
        
        # For now, returning dummy high score to prove architecture flow
        return 8