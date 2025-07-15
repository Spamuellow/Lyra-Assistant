"""
Screen Capture Handler
Captures and processes screen content for vision analysis
"""

import cv2
import numpy as np
import mss
import threading
import queue
import time
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple
from PIL import Image
import base64
import io

from config import SCREEN_CAPTURE, DEBUG

logger = logging.getLogger(__name__)

class ScreenCaptureHandler:
    """Handles screen capture and processing for vision analysis"""
    
    def __init__(self):
        self.sct = mss.mss()
        self.is_capturing = False
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=5)  # Limit queue size
        
        # Capture settings
        self.monitor = SCREEN_CAPTURE["monitor"]
        self.fps = SCREEN_CAPTURE["fps"]
        self.compression_quality = SCREEN_CAPTURE["compression_quality"]
        self.max_width = SCREEN_CAPTURE["max_width"]
        self.max_height = SCREEN_CAPTURE["max_height"]
        
        # Get monitor info
        self.monitor_info = self._get_monitor_info()
        
        # Callbacks
        self.on_frame_captured = None
        self.on_significant_change = None
        
        # Change detection
        self.last_frame = None
        self.change_threshold = 0.05  # 5% change threshold
        
        logger.info(f"ScreenCaptureHandler initialized for monitor {self.monitor}")
        logger.info(f"Monitor resolution: {self.monitor_info}")
    
    def _get_monitor_info(self) -> Dict[str, Any]:
        """Get information about the selected monitor"""
        try:
            monitors = self.sct.monitors
            if self.monitor < len(monitors):
                return monitors[self.monitor]
            else:
                logger.warning(f"Monitor {self.monitor} not found, using primary monitor")
                return monitors[1]  # Primary monitor
        except Exception as e:
            logger.error(f"Failed to get monitor info: {e}")
            return {"top": 0, "left": 0, "width": 1920, "height": 1080}
    
    def set_callbacks(self, 
                     on_frame_captured: Optional[Callable[[np.ndarray], None]] = None,
                     on_significant_change: Optional[Callable[[np.ndarray], None]] = None):
        """Set callback functions for capture events"""
        self.on_frame_captured = on_frame_captured
        self.on_significant_change = on_significant_change
    
    def start_capture(self) -> bool:
        """Start continuous screen capture"""
        if self.is_capturing:
            logger.warning("Already capturing")
            return True
        
        try:
            logger.info("Starting screen capture...")
            self.is_capturing = True
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start capture: {e}")
            return False
    
    def stop_capture(self):
        """Stop screen capture"""
        if not self.is_capturing:
            return
        
        logger.info("Stopping screen capture...")
        self.is_capturing = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
    
    def _capture_loop(self):
        """Main capture loop"""
        frame_interval = 1.0 / self.fps
        
        try:
            while self.is_capturing:
                start_time = time.time()
                
                # Capture frame
                frame = self.capture_frame()
                
                if frame is not None:
                    # Add to queue (non-blocking)
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Remove oldest frame if queue is full
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except queue.Empty:
                            pass
                    
                    # Call callbacks
                    if self.on_frame_captured:
                        self.on_frame_captured(frame)
                    
                    # Check for significant changes
                    if self._detect_significant_change(frame):
                        if self.on_significant_change:
                            self.on_significant_change(frame)
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in capture loop: {e}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the screen"""
        try:
            # Capture screenshot
            screenshot = self.sct.grab(self.monitor_info)
            
            # Convert to numpy array
            frame = np.array(screenshot)
            
            # Convert from BGRA to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            
            # Resize if needed
            frame = self._resize_frame(frame)
            
            if DEBUG["log_timings"]:
                logger.debug(f"Frame captured: {frame.shape}")
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if it exceeds maximum dimensions"""
        height, width = frame.shape[:2]
        
        if width <= self.max_width and height <= self.max_height:
            return frame
        
        # Calculate scaling factor
        scale_width = self.max_width / width
        scale_height = self.max_height / height
        scale = min(scale_width, scale_height)
        
        # Resize
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        logger.debug(f"Frame resized from {width}x{height} to {new_width}x{new_height}")
        return resized
    
    def _detect_significant_change(self, current_frame: np.ndarray) -> bool:
        """Detect if there's a significant change from the last frame"""
        if self.last_frame is None:
            self.last_frame = current_frame.copy()
            return True
        
        # Calculate frame difference
        diff = cv2.absdiff(self.last_frame, current_frame)
        
        # Convert to grayscale for easier processing
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        
        # Calculate percentage of changed pixels
        total_pixels = gray_diff.size
        changed_pixels = np.sum(gray_diff > 30)  # Threshold for change
        change_percentage = changed_pixels / total_pixels
        
        # Update last frame
        self.last_frame = current_frame.copy()
        
        is_significant = change_percentage > self.change_threshold
        
        if DEBUG["log_timings"] and is_significant:
            logger.debug(f"Significant change detected: {change_percentage:.2%}")
        
        return is_significant
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def capture_region(self, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """Capture a specific region of the screen"""
        try:
            # Define region
            region = {
                "top": y,
                "left": x,
                "width": width,
                "height": height
            }
            
            # Capture region
            screenshot = self.sct.grab(region)
            
            # Convert to numpy array
            frame = np.array(screenshot)
            
            # Convert from BGRA to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            
            logger.debug(f"Region captured: {width}x{height} at ({x}, {y})")
            return frame
            
        except Exception as e:
            logger.error(f"Failed to capture region: {e}")
            return None
    
    def save_frame(self, frame: np.ndarray, filepath: Path) -> bool:
        """Save frame to file"""
        try:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Save image
            cv2.imwrite(str(filepath), bgr_frame)
            
            logger.info(f"Frame saved to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            return False
    
    def frame_to_base64(self, frame: np.ndarray, format: str = "JPEG") -> Optional[str]:
        """Convert frame to base64 string"""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Convert to bytes
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format, quality=self.compression_quality)
            
            # Convert to base64
            base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return base64_string
            
        except Exception as e:
            logger.error(f"Failed to convert frame to base64: {e}")
            return None
    
    def frame_to_bytes(self, frame: np.ndarray, format: str = "JPEG") -> Optional[bytes]:
        """Convert frame to bytes"""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Convert to bytes
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format, quality=self.compression_quality)
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to convert frame to bytes: {e}")
            return None
    
    def get_screen_info(self) -> Dict[str, Any]:
        """Get screen information"""
        return {
            "monitor": self.monitor,
            "resolution": (self.monitor_info["width"], self.monitor_info["height"]),
            "capture_fps": self.fps,
            "max_size": (self.max_width, self.max_height),
            "is_capturing": self.is_capturing
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.stop_capture()
            if self.sct:
                self.sct.close()
            logger.info("Screen capture handler cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Test function
def test_screen_capture():
    """Test the screen capture handler"""
    handler = ScreenCaptureHandler()
    
    def on_frame_captured(frame):
        print(f"Frame captured: {frame.shape}")
    
    def on_significant_change(frame):
        print("Significant change detected!")
    
    handler.set_callbacks(
        on_frame_captured=on_frame_captured,
        on_significant_change=on_significant_change
    )
    
    print("Starting screen capture test...")
    handler.start_capture()
    
    try:
        time.sleep(5)  # Capture for 5 seconds
        
        # Test single frame capture
        frame = handler.capture_frame()
        if frame is not None:
            print(f"Single frame captured: {frame.shape}")
            
            # Test saving frame
            test_path = Path("test_screenshot.jpg")
            if handler.save_frame(frame, test_path):
                print(f"Frame saved to: {test_path}")
            
            # Test base64 conversion
            base64_str = handler.frame_to_base64(frame)
            if base64_str:
                print(f"Base64 conversion successful: {len(base64_str)} characters")
        
    except KeyboardInterrupt:
        pass
    
    handler.stop_capture()
    handler.cleanup()
    print("Screen capture test completed")
    return True

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_screen_capture()