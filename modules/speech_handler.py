"""
Speech Handler
Handles speech-to-text using Faster-Whisper and integrates with Chatterbox TTS
"""

import torch
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import logging
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
import wave
import tempfile

# Import faster-whisper
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Faster-Whisper not found. Please install it with: pip install faster-whisper")
    raise

from config import MODELS, AUDIO, VOICE_ACTIVATION, DEBUG
from modules.chatterbox_handler import ChatterboxHandler

logger = logging.getLogger(__name__)

class SpeechHandler:
    """Handles speech-to-text and text-to-speech operations"""
    
    def __init__(self):
        self.stt_model = None
        self.tts_handler = ChatterboxHandler()
        self.is_initialized = False
        
        # Audio settings
        self.sample_rate = AUDIO["sample_rate"]
        self.channels = AUDIO["channels"]
        self.chunk_size = AUDIO["chunk_size"]
        
        # Recording state
        self.is_recording = False
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        
        # Voice activation
        self.voice_threshold = VOICE_ACTIVATION["threshold"]
        self.silence_timeout = VOICE_ACTIVATION["timeout"]
        
        # Callbacks
        self.on_speech_detected = None
        self.on_speech_recognized = None
        self.on_silence_detected = None
        
        logger.info("SpeechHandler initialized")
    
    def initialize(self) -> bool:
        """Initialize speech recognition and TTS models"""
        try:
            logger.info("Initializing speech models...")
            
            # Initialize Faster-Whisper STT
            self._initialize_stt()
            
            # Initialize Chatterbox TTS
            if not self.tts_handler.initialize():
                logger.error("Failed to initialize TTS handler")
                return False
            
            self.is_initialized = True
            logger.info("Speech handler initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize speech handler: {e}")
            return False
    
    def _initialize_stt(self):
        """Initialize Faster-Whisper STT model"""
        try:
            logger.info("Loading Faster-Whisper model...")
            
            self.stt_model = WhisperModel(
                model_size_or_path=MODELS["stt"]["model_size"],
                device=MODELS["stt"]["device"],
                compute_type=MODELS["stt"]["compute_type"],
                cpu_threads=4,
                num_workers=1
            )
            
            logger.info("Faster-Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper: {e}")
            raise
    
    def set_callbacks(self, 
                     on_speech_detected: Optional[Callable] = None,
                     on_speech_recognized: Optional[Callable[[str], None]] = None,
                     on_silence_detected: Optional[Callable] = None):
        """Set callback functions for speech events"""
        self.on_speech_detected = on_speech_detected
        self.on_speech_recognized = on_speech_recognized
        self.on_silence_detected = on_silence_detected
    
    def start_listening(self) -> bool:
        """Start continuous listening for speech"""
        if not self.is_initialized:
            logger.error("Speech handler not initialized")
            return False
        
        if self.is_listening:
            logger.warning("Already listening")
            return True
        
        try:
            logger.info("Starting continuous listening...")
            self.is_listening = True
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start listening: {e}")
            return False
    
    def stop_listening(self):
        """Stop continuous listening"""
        if not self.is_listening:
            return
        
        logger.info("Stopping continuous listening...")
        self.is_listening = False
        
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
    
    def _recording_loop(self):
        """Main recording loop for continuous listening"""
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            ):
                logger.info("Audio input stream started")
                
                while self.is_listening:
                    self._process_audio_queue()
                    time.sleep(0.01)  # Small delay to prevent CPU spinning
                    
        except Exception as e:
            logger.error(f"Error in recording loop: {e}")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Add audio data to queue
        self.audio_queue.put(indata.copy())
    
    def _process_audio_queue(self):
        """Process audio data from the queue"""
        if self.audio_queue.empty():
            return
        
        # Collect audio chunks
        audio_chunks = []
        while not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                audio_chunks.append(chunk)
            except queue.Empty:
                break
        
        if not audio_chunks:
            return
        
        # Combine chunks
        audio_data = np.concatenate(audio_chunks, axis=0)
        
        # Check for speech activity
        if self._detect_speech_activity(audio_data):
            self._handle_speech_detection(audio_data)
    
    def _detect_speech_activity(self, audio_data: np.ndarray) -> bool:
        """Detect if there's speech activity in the audio"""
        # Simple energy-based voice activity detection
        energy = np.sqrt(np.mean(audio_data ** 2))
        return energy > self.voice_threshold
    
    def _handle_speech_detection(self, audio_data: np.ndarray):
        """Handle detected speech"""
        if self.on_speech_detected:
            self.on_speech_detected()
        
        # Record until silence
        full_audio = self._record_until_silence(audio_data)
        
        if full_audio is not None and len(full_audio) > 0:
            # Transcribe the audio
            text = self.transcribe_audio(full_audio)
            
            if text and self.on_speech_recognized:
                self.on_speech_recognized(text)
    
    def _record_until_silence(self, initial_audio: np.ndarray) -> Optional[np.ndarray]:
        """Record audio until silence is detected"""
        audio_buffer = [initial_audio]
        silence_duration = 0.0
        chunk_duration = self.chunk_size / self.sample_rate
        
        logger.debug("Recording speech...")
        
        while self.is_listening:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer.append(chunk)
                
                # Check for silence
                if self._detect_speech_activity(chunk):
                    silence_duration = 0.0
                else:
                    silence_duration += chunk_duration
                    
                    if silence_duration >= self.silence_timeout:
                        logger.debug("Silence detected, stopping recording")
                        break
                        
            except queue.Empty:
                silence_duration += 0.1
                if silence_duration >= self.silence_timeout:
                    break
        
        if len(audio_buffer) > 0:
            return np.concatenate(audio_buffer, axis=0)
        return None
    
    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio data to text"""
        if not self.is_initialized:
            logger.error("Speech handler not initialized")
            return None
        
        try:
            start_time = time.time()
            
            # Convert to mono if needed
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Transcribe using Faster-Whisper
            segments, info = self.stt_model.transcribe(
                audio_data,
                beam_size=5,
                language="en",  # Can be made configurable
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Combine all segments
            text = ""
            for segment in segments:
                text += segment.text
            
            text = text.strip()
            
            if DEBUG["log_timings"]:
                transcription_time = time.time() - start_time
                logger.debug(f"Transcription took {transcription_time:.2f}s")
            
            if text:
                logger.info(f"Transcribed: {text}")
                return text
            else:
                logger.debug("No speech detected in audio")
                return None
                
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            return None
    
    def transcribe_file(self, audio_file_path: Path) -> Optional[str]:
        """Transcribe audio file to text"""
        if not self.is_initialized:
            logger.error("Speech handler not initialized")
            return None
        
        try:
            logger.info(f"Transcribing file: {audio_file_path}")
            
            segments, info = self.stt_model.transcribe(
                str(audio_file_path),
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True
            )
            
            text = ""
            for segment in segments:
                text += segment.text
            
            text = text.strip()
            logger.info(f"File transcribed: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Failed to transcribe file: {e}")
            return None
    
    def speak(self, text: str, emotion: str = "neutral", blocking: bool = False) -> bool:
        """Speak text using TTS"""
        return self.tts_handler.speak(text, emotion, blocking)
    
    def stop_speaking(self):
        """Stop current speech"""
        self.tts_handler.stop_speech()
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self.tts_handler.is_speaking_now()
    
    def clone_voice(self, reference_audio_path: Path, voice_name: str = "default") -> bool:
        """Clone voice from reference audio"""
        return self.tts_handler.clone_voice(reference_audio_path, voice_name)
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.stop_listening()
            self.tts_handler.cleanup()
            logger.info("Speech handler cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Test function
def test_speech_handler():
    """Test the speech handler"""
    handler = SpeechHandler()
    
    def on_speech_recognized(text: str):
        print(f"You said: {text}")
        handler.speak(f"You said: {text}")
    
    handler.set_callbacks(on_speech_recognized=on_speech_recognized)
    
    if not handler.initialize():
        print("Failed to initialize speech handler")
        return False
    
    print("Speech handler test - speak something...")
    handler.start_listening()
    
    try:
        time.sleep(10)  # Listen for 10 seconds
    except KeyboardInterrupt:
        pass
    
    handler.stop_listening()
    handler.cleanup()
    print("Speech handler test completed")
    return True

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_speech_handler()
