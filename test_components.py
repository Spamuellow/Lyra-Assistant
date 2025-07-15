"""
Component Test Suite
Tests all AI Assistant components individually
"""

import sys
import logging
import time
import torch
from pathlib import Path

# Add project root to path
project_root = Path.home() / "ai-assistant"
sys.path.insert(0, str(project_root))

from modules.chatterbox_handler import ChatterboxHandler
from modules.speech_handler import SpeechHandler
from modules.screen_capture import ScreenCaptureHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_system_requirements():
    """Test system requirements"""
    print("=" * 60)
    print("SYSTEM REQUIREMENTS TEST")
    print("=" * 60)
    
    # Check CUDA
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    
    # Check project structure
    print(f"Project Root: {project_root}")
    print(f"Project Root Exists: {project_root.exists()}")
    
    modules_dir = project_root / "modules"
    print(f"Modules Directory: {modules_dir.exists()}")
    
    print("\n✓ System requirements check completed\n")

def test_chatterbox_tts():
    """Test Chatterbox TTS functionality"""
    print("=" * 60)
    print("CHATTERBOX TTS TEST")
    print("=" * 60)
    
    try:
        handler = ChatterboxHandler()
        
        print("Initializing Chatterbox TTS...")
        if handler.initialize():
            print("✓ Chatterbox TTS initialized successfully")
            
            # Test basic speech
            test_text = "Hello! This is a test of the Chatterbox TTS system. If you can hear this, the test is working correctly."
            print(f"Testing speech: {test_text}")
            
            if handler.speak(test_text, blocking=True):
                print("✓ Speech synthesis successful")
            else:
                print("✗ Speech synthesis failed")
        else:
            print("✗ Failed to initialize Chatterbox TTS")
            return False
        
        handler.cleanup()
        print("✓ Chatterbox TTS test completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Chatterbox TTS test failed: {e}")
        return False

def test_speech_recognition():
    """Test Faster-Whisper speech recognition"""
    print("=" * 60)
    print("SPEECH RECOGNITION TEST")
    print("=" * 60)
    
    try:
        handler = SpeechHandler()
        
        print("Initializing Speech Handler...")
        if handler.initialize():
            print("✓ Speech Handler initialized successfully")
            
            # Test if we can access the STT model
            if handler.stt_model:
                print("✓ Faster-Whisper model loaded")
                
                # Test with a simple audio file (if available)
                # For now, just confirm initialization
                print("✓ STT model ready for transcription")
            else:
                print("✗ STT model not loaded")
                return False
        else:
            print("✗ Failed to initialize Speech Handler")
            return False
        
        handler.cleanup()
        print("✓ Speech recognition test completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Speech recognition test failed: {e}")
        return False

def test_screen_capture():
    """Test screen capture functionality"""
    print("=" * 60)
    print("SCREEN CAPTURE TEST")
    print("=" * 60)
    
    try:
        handler = ScreenCaptureHandler()
        
        print("Testing screen capture...")
        
        # Test single frame capture
        frame = handler.capture_frame()
        if frame is not None:
            print(f"✓ Frame captured successfully: {frame.shape}")
            
            # Test frame conversion
            base64_str = handler.frame_to_base64(frame)
            if base64_str:
                print(f"✓ Base64 conversion successful: {len(base64_str)} characters")
            
            # Test saving frame
            test_path = project_root / "test_screenshot.jpg"
            if handler.save_frame(frame, test_path):
                print(f"✓ Frame saved to: {test_path}")
        else:
            print("✗ Frame capture failed")
            return False
        
        # Test continuous capture for a short time
        print("Testing continuous capture (3 seconds)...")
        capture_count = 0
        
        def on_frame_captured(frame):
            nonlocal capture_count
            capture_count += 1
        
        handler.set_callbacks(on_frame_captured=on_frame_captured)
        handler.start_capture()
        time.sleep(3)
        handler.stop_capture()
        
        print(f"✓ Captured {capture_count} frames in 3 seconds")
        
        handler.cleanup()
        print("✓ Screen capture test completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Screen capture test failed: {e}")
        return False

def test_integrated_speech():
    """Test integrated speech (STT + TTS)"""
    print("=" * 60)
    print("INTEGRATED SPEECH TEST")
    print("=" * 60)
    
    try:
        handler = SpeechHandler()
        
        print("Initializing integrated speech handler...")
        if handler.initialize():
            print("✓ Integrated speech handler initialized")
            
            # Test TTS
            test_text = "This is a test of the integrated speech system."
            print(f"Testing TTS: {test_text}")
            
            if handler.speak(test_text, blocking=True):
                print("✓ TTS working in integrated handler")
            else:
                print("✗ TTS failed in integrated handler")
                return False
            
            print("✓ Integrated speech test completed")
        else:
            print("✗ Failed to initialize integrated speech handler")
            return False
        
        handler.cleanup()
        print("✓ Integrated speech test completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Integrated speech test failed: {e}")
        return False

def test_memory_usage():
    """Test VRAM and memory usage"""
    print("=" * 60)
    print("MEMORY USAGE TEST")
    print("=" * 60)
    
    if torch.cuda.is_available():
        # Check initial VRAM usage
        initial_vram = torch.cuda.memory_allocated(0) / 1024**3
        print(f"Initial VRAM usage: {initial_vram:.2f} GB")
        
        # Test each component's memory usage
        try:
            # Test Chatterbox TTS memory
            print("Testing Chatterbox TTS memory usage...")
            tts_handler = ChatterboxHandler()
            tts_handler.initialize()
            
            tts_vram = torch.cuda.memory_allocated(0) / 1024**3
            print(f"VRAM after TTS init: {tts_vram:.2f} GB (+{tts_vram - initial_vram:.2f} GB)")
            
            tts_handler.cleanup()
            torch.cuda.empty_cache()
            
            # Test Speech Handler memory
            print("Testing Speech Handler memory usage...")
            speech_handler = SpeechHandler()
            speech_handler.initialize()
            
            speech_vram = torch.cuda.memory_allocated(0) / 1024**3
            print(f"VRAM after Speech init: {speech_vram:.2f} GB")
            
            speech_handler.cleanup()
            torch.cuda.empty_cache()
            
            final_vram = torch.cuda.memory_allocated(0) / 1024**3
            print(f"Final VRAM usage: {final_vram:.2f} GB")
            
            print("✓ Memory usage test completed")
            
        except Exception as e:
            print(f"✗ Memory usage test failed: {e}")
            return False
    else:
        print("CUDA not available, skipping VRAM test")
    
    print("✓ Memory usage test completed\n")
    return True

def run_all_tests():
    """Run all component tests"""
    print("AI ASSISTANT COMPONENT TEST SUITE")
    print("=" * 60)
    
    test_results = {
        "System Requirements": test_system_requirements,
        "Chatterbox TTS": test_chatterbox_tts,
        "Speech Recognition": test_speech_recognition,
        "Screen Capture": test_screen_capture,
        "Integrated Speech": test_integrated_speech,
        "Memory Usage": test_memory_usage,
    }
    
    results = {}
    
    for test_name, test_func in test_results.items():
        try:
            print(f"\nRunning {test_name} test...")
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed! Ready to build the main application.")
    else:
        print(f"\n⚠️  {failed} tests failed. Please fix issues before proceeding.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
