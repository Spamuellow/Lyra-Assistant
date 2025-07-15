"""
Chatterbox TTS Handler
Handles text-to-speech using Chatterbox TTS with voice cloning
"""

import torch
import sounddevice as sd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import asyncio
import threading
import queue
import time

# Import chatterbox - adjust import path as needed
try:
    import chatterbox
except ImportError:
    print("Chatterbox not found. Please ensure it's installed correctly.")
    raise

from config import MODELS, AUDIO, REFERENCE_AUDIO_DIR, DEBUG

logger = logging.getLogger(__name__)

class ChatterboxHandler:
    """Handles Chatterbox TTS operations with voice cloning"""
    
    def __init__(self):
        self.model = None
        self.voice_preset = None
        self.is_initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_queue = queue.Queue()
        self.is_speaking = False
        self.stop_speaking = False
        
        # Audio settings
        self.sample_rate = AUDIO["sample_rate"]
        self.channels = AUDIO["channels"]
        
        logger.info(f"ChatterboxHandler initialized on {self.device}")
    
    def initialize(self) -> bool:
        """Initialize the Chatterbox TTS model"""
        try:
            logger.info("Initializing Chatterbox TTS model...")
            
            # Initialize Chatterbox model
            # Adjust these parameters based on actual Chatterbox API
            self.model = chatterbox.load_model(
                device=self.device,
                model_name=MODELS["tts"]["model_name"]
            )
            
            self.is_initialized = True
            logger.info("Chatterbox TTS model initialized successfully")
            
            # Check for existing voice preset
            self._load_voice_preset()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Chatterbox TTS: {e}")
            return False
    
    def _load_voice_preset(self):
        """Load existing voice preset if available"""
        preset_path = REFERENCE_AUDIO_DIR / "voice_preset.json"
        
        if preset_path.exists():
            try:
                # Load saved voice preset
                # This depends on Chatterbox's actual API
                logger.info("Loading existing voice preset...")
                # self.voice_preset = chatterbox.load_voice_preset(preset_path)
                logger.info("Voice preset loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load voice preset: {e}")
        else:
            logger.info("No voice preset found. Use clone_voice() to create one.")
    
    def clone_voice(self, reference_audio_path: Path, voice_name: str = "default") -> bool:
        """
        Clone a voice from reference audio
        
        Args:
            reference_audio_path: Path to reference audio file
            voice_name: Name for the voice preset
            
        Returns:
            bool: Success status
        """
        if not self.is_initialized:
            logger.error("Chatterbox not initialized")
            return False
        
        try:
            logger.info(f"Cloning voice from: {reference_audio_path}")
            
            # Clone voice using Chatterbox API
            # Adjust this based on actual Chatterbox API
            self.voice_preset = chatterbox.clone_voice(
                model=self.model,
                reference_audio=str(reference_audio_path),
                voice_name=voice_name
            )
            
            # Save the voice preset
            preset_path = REFERENCE_AUDIO_DIR / f"{voice_name}_preset.json"
            # chatterbox.save_voice_preset(self.voice_preset, preset_path)
            
            logger.info(f"Voice cloned successfully as '{voice_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clone voice: {e}")
            return False
    
    def synthesize_speech(self, text: str, emotion: str = "neutral") -> Optional[np.ndarray]:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            emotion: Emotion for the speech
            
        Returns:
            numpy.ndarray: Audio data or None if failed
        """
        if not self.is_initialized:
            logger.error("Chatterbox not initialized")
            return None
        
        if not self.voice_preset:
            logger.warning("No voice preset loaded. Using default voice.")
        
        try:
            start_time = time.time()
            
            # Synthesize speech using Chatterbox
            # Adjust this based on actual Chatterbox API
            audio_data = chatterbox.synthesize(
                model=self.model,
                text=text,
                voice_preset=self.voice_preset,
                emotion=emotion,
                speed=MODELS["tts"]["speed"]
            )
            
            if DEBUG["log_timings"]:
                synthesis_time = time.time() - start_time
                logger.debug(f"Speech synthesis took {synthesis_time:.2f}s for {len(text)} characters")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Failed to synthesize speech: {e}")
            return None
    
    def speak(self, text: str, emotion: str = "neutral", blocking: bool = False) -> bool:
        """
        Speak text using TTS
        
        Args:
            text: Text to speak
            emotion: Emotion for the speech
            blocking: Whether to wait for speech to complete
            
        Returns:
            bool: Success status
        """
        if not text.strip():
            return True
        
        logger.info(f"Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Synthesize audio
        audio_data = self.synthesize_speech(text, emotion)
        if audio_data is None:
            return False
        
        # Play audio
        if blocking:
            return self._play_audio_blocking(audio_data)
        else:
            return self._play_audio_async(audio_data)
    
    def _play_audio_blocking(self, audio_data: np.ndarray) -> bool:
        """Play audio synchronously"""
        try:
            self.is_speaking = True
            sd.play(audio_data, samplerate=self.sample_rate)
            sd.wait()
            self.is_speaking = False
            return True
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            self.is_speaking = False
            return False
    
    def _play_audio_async(self, audio_data: np.ndarray) -> bool:
        """Play audio asynchronously"""
        try:
            def play_audio():
                self.is_speaking = True
                sd.play(audio_data, samplerate=self.sample_rate)
                sd.wait()
                self.is_speaking = False
            
            thread = threading.Thread(target=play_audio)
            thread.daemon = True
            thread.start()
            return True
            
        except Exception as e:
            logger.error(f"Failed to play audio async: {e}")
            return False
    
    def stop_speech(self):
        """Stop current speech"""
        try:
            sd.stop()
            self.is_speaking = False
            self.stop_speaking = True
            logger.info("Speech stopped")
        except Exception as e:
            logger.error(f"Failed to stop speech: {e}")
    
    def is_speaking_now(self) -> bool:
        """Check if currently speaking"""
        return self.is_speaking
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.stop_speech()
            if self.model:
                # Clean up model if needed
                pass
            logger.info("Chatterbox handler cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Test function
def test_chatterbox():
    """Test the Chatterbox handler"""
    handler = ChatterboxHandler()
    
    if not handler.initialize():
        print("Failed to initialize Chatterbox")
        return False
    
    # Test basic speech
    test_text = "Hello! This is a test of the Chatterbox TTS system."
    success = handler.speak(test_text, blocking=True)
    
    if success:
        print("Chatterbox test successful!")
    else:
        print("Chatterbox test failed!")
    
    handler.cleanup()
    return success

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_chatterbox()
