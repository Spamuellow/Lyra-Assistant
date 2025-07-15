"""
Configuration settings for AI Assistant
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path.home() / "ai-assistant"
REFERENCE_AUDIO_DIR = PROJECT_ROOT / "reference_audio"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Hardware settings
GPU_DEVICE = "cuda"
MAX_VRAM_GB = 20  # Leave 4GB buffer on RTX 4090

# VRAM allocation (in GB)
VRAM_ALLOCATION = {
    "llm": 10,        # Main language model
    "vision": 6,      # Vision model for screen analysis
    "tts": 2,         # Chatterbox TTS
    "stt": 2,         # Faster-Whisper STT
}

# Model configurations
MODELS = {
    "llm": {
        "name": "to_be_selected",  # 13B GGUF model
        "max_tokens": 4096,
        "temperature": 0.8,
        "top_p": 0.9,
    },
    "vision": {
        "name": "to_be_selected",  # Qwen2.5-VL 7B, Pixtral 12B, or Phi-3.5 Vision
        "max_tokens": 1024,
    },
    "stt": {
        "model_size": "large-v3",  # Faster-Whisper model
        "device": "cuda",
        "compute_type": "float16",
    },
    "tts": {
        "model_name": "chatterbox",
        "voice_preset": "default",  # Will be updated after voice cloning
        "speed": 1.0,
        "emotion": "neutral",
    }
}

# Audio settings
AUDIO = {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_size": 1024,
    "format": "float32",
    "input_device": None,  # Auto-select
    "output_device": None,  # Auto-select
}

# Screen capture settings
SCREEN_CAPTURE = {
    "monitor": 1,  # Primary monitor
    "fps": 2,      # Capture rate for analysis
    "compression_quality": 85,
    "max_width": 1920,
    "max_height": 1080,
}

# Assistant behavior
ASSISTANT = {
    "name": "Assistant",
    "personality": "friendly, helpful, and engaging",
    "roleplay_mode": True,
    "uncensored": True,
    "context_window": 8192,
    "memory_length": 50,  # Number of previous interactions to remember
}

# Logging
LOGGING = {
    "level": "INFO",
    "file": LOGS_DIR / "assistant.log",
    "max_size": "10MB",
    "backup_count": 5,
}

# Development settings
DEBUG = {
    "enabled": True,
    "verbose_gpu": False,
    "log_timings": True,
    "test_mode": False,
}

# Voice activation
VOICE_ACTIVATION = {
    "enabled": True,
    "threshold": 0.5,
    "timeout": 5.0,  # Seconds of silence before stopping
    "wake_word": None,  # Optional wake word
}

# Create directories if they don't exist
for directory in [REFERENCE_AUDIO_DIR, LOGS_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)
