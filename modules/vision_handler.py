"""
Vision Handler
Handles vision model operations for screen analysis
"""

import torch
import requests
import json
import base64
import logging
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
from PIL import Image
import io

from config import MODELS, DEBUG

logger = logging.getLogger(__name__)

class VisionHandler:
    """Handles vision model operations for screen analysis"""
    
    def __init__(self):
        self.model_name = None
        self.is_initialized = False
        self.ollama_url = "http://localhost:11434"
        
        # Model settings
        self.max_tokens = MODELS["vision"]["max_tokens"]
        
        # Vision analysis settings
        self.analysis_prompts = {
            "general": "Describe what you see in this screen capture. Focus on the main content, applications, and any notable activities.",
            "detailed": "Provide a detailed analysis of this screen capture. Include UI elements, text content, applications running, and user activities.",
            "gaming": "Analyze this gaming screen capture. Identify the game, current activity, UI elements, and any relevant gaming information.",
            "application": "Analyze this application screen capture. Identify the application, current function, and what the user appears to be doing.",
            "change_detection": "Compare this screen to previous context and identify what has changed or what new activity is happening."
        }
        
        logger.info("VisionHandler initialized")
    
    def initialize(self, model_name: str = None) -> bool:
        """Initialize the vision handler with a specific model"""
        try:
            logger.info("Initializing Vision handler...")
            
            # Check if Ollama is running
            if not self._check_ollama_status():
                logger.error("Ollama is not running. Please start Ollama first.")
                return False
            
            # Set model name
            if model_name:
                self.model_name = model_name
            else:
                # Auto-detect available vision models
                available_models = self._get_available_vision_models()
                if not available_models:
                    logger.error("No vision models available in Ollama")
                    return False
                
                # Select first available vision model
                self.model_name = available_models[0]
            
            logger.info(f"Using vision model: {self.model_name}")
            
            # Test model with a simple image
            if self._test_model():
                logger.info("Vision model test successful")
                self.is_initialized = True
                return True
            else:
                logger.error("Vision model test failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize vision handler: {e}")
            return False
    
    def _check_ollama_status(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _get_available_vision_models(self) -> List[str]:
        """Get list of available vision models from Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = []
                
                # Filter for vision models (models that support images)
                vision_model_names = [
                    "llava", "bakllava", "llava-llama3", "llava-phi3", 
                    "moondream", "qwen2-vl", "pixtral", "phi3-vision"
                ]
                
                for model in data.get("models", []):
                    model_name = model["name"].lower()
                    if any(vision_name in model_name for vision_name in vision_model_names):
                        models.append(model["name"])
                
                logger.info(f"Available vision models: {models}")
                return models
            return []
        except Exception as e:
            logger.error(f"Failed to get available vision models: {e}")
            return []
    
    def _test_model(self) -> bool:
        """Test the vision model with a simple prompt"""
        try:
            # Create a simple test image (solid color)
            test_image = Image.new('RGB', (100, 100), color='red')
            
            # Convert to base64
            buffer = io.BytesIO()
            test_image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Test with model
            response = self._analyze_image(image_base64, "What color is this image?")
            return response is not None
            
        except Exception as e:
            logger.error(f"Vision model test failed: {e}")
            return False
    
    def analyze_screen(self, image_data: np.ndarray, analysis_type: str = "general") -> Optional[str]:
        """
        Analyze a screen capture
        
        Args:
            image_data: Screen capture as numpy array
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis result or None if failed
        """
        if not self.is_initialized:
            logger.error("Vision handler not initialized")
            return None
        
        try:
            # Convert numpy array to base64
            image_base64 = self._numpy_to_base64(image_data)
            if not image_base64:
                logger.error("Failed to convert image to base64")
                return None
            
            # Get analysis prompt
            prompt = self.analysis_prompts.get(analysis_type, self.analysis_prompts["general"])
            
            # Analyze image
            result = self._analyze_image(image_base64, prompt)
            
            if result:
                logger.info(f"Screen analysis completed: {analysis_type}")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze screen: {e}")
            return None
    
    def analyze_image_file(self, image_path: Path, prompt: str = None) -> Optional[str]:
        """
        Analyze an image file
        
        Args:
            image_path: Path to image file
            prompt: Custom prompt for analysis
            
        Returns:
            Analysis result or None if failed
        """
        if not self.is_initialized:
            logger.error("Vision handler not initialized")
            return None
        
        try:
            # Read and convert image to base64
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Use custom prompt or default
            if not prompt:
                prompt = self.analysis_prompts["general"]
            
            # Analyze image
            result = self._analyze_image(image_base64, prompt)
            
            if result:
                logger.info(f"Image file analysis completed: {image_path}")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze image file: {e}")
            return None
    
    def _analyze_image(self, image_base64: str, prompt: str) -> Optional[str]:
        """Analyze image using Ollama vision model"""
        try:
            start_time = time.time()
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_base64],
                "options": {
                    "num_predict": self.max_tokens,
                },
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # Vision models can be slower
            )
            
            if response.status_code == 200:
                data = response.json()
                analysis_result = data.get("response", "").strip()
                
                if DEBUG["log_timings"]:
                    analysis_time = time.time() - start_time
                    logger.debug(f"Vision analysis took {analysis_time:.2f}s")
                
                return analysis_result
            else:
                logger.error(f"Ollama vision API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to analyze image: {e}")
            return None
    
    def _numpy_to_base64(self, image_array: np.ndarray) -> Optional[str]:
        """Convert numpy array to base64 string"""
        try:
            # Convert to PIL Image
            if image_array.dtype != np.uint8:
                # Normalize to 0-255 if needed
                image_array = (image_array * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image_array)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Failed to convert numpy array to base64: {e}")
            return None
    
    def detect_changes(self, current_image: np.ndarray, previous_analysis: str = None) -> Optional[str]:
        """
        Detect changes in screen content
        
        Args:
            current_image: Current screen capture
            previous_analysis: Previous analysis for comparison
            
        Returns:
            Change detection result
        """
        if not self.is_initialized:
            logger.error("Vision handler not initialized")
            return None
        
        try:
            # Convert to base64
            image_base64 = self._numpy_to_base64(current_image)
            if not image_base64:
                return None
            
            # Build change detection prompt
            prompt = self.analysis_prompts["change_detection"]
            if previous_analysis:
                prompt += f"\n\nPrevious analysis for comparison: {previous_analysis}"
            
            # Analyze for changes
            result = self._analyze_image(image_base64, prompt)
            
            if result:
                logger.info("Change detection completed")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect changes: {e}")
            return None
    
    def identify_application(self, image_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Identify the main application in the screen capture
        
        Args:
            image_data: Screen capture as numpy array
            
        Returns:
            Application information or None if failed
        """
        if not self.is_initialized:
            return None
        
        try:
            # Convert to base64
            image_base64 = self._numpy_to_base64(image_data)
            if not image_base64:
                return None
            
            # Application identification prompt
            prompt = """Identify the main application or software visible in this screen capture. 
            Provide the following information in a structured format:
            - Application name
            - Type (browser, game, editor, etc.)
            - Current activity or function
            - Notable UI elements
            
            Format your response as a clear analysis."""
            
            # Analyze image
            result = self._analyze_image(image_base64, prompt)
            
            if result:
                # Parse result into structured format
                return {
                    "raw_analysis": result,
                    "timestamp": time.time()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to identify application: {e}")
            return None
    
    def get_text_content(self, image_data: np.ndarray) -> Optional[str]:
        """
        Extract text content from screen capture
        
        Args:
            image_data: Screen capture as numpy array
            
        Returns:
            Extracted text or None if failed
        """
        if not self.is_initialized:
            return None
        
        try:
            # Convert to base64
            image_base64 = self._numpy_to_base64(image_data)
            if not image_base64:
                return None
            
            # Text extraction prompt
            prompt = """Extract and list all readable text from this screen capture. 
            Focus on:
            - Main text content
            - UI labels and buttons
            - Menu items
            - Any visible text
            
            Present the text in a clear, organized format."""
            
            # Analyze image
            result = self._analyze_image(image_base64, prompt)
            
            if result:
                logger.info("Text extraction completed")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            return None
    
    def add_custom_prompt(self, name: str, prompt: str):
        """Add a custom analysis prompt"""
        self.analysis_prompts[name] = prompt
        logger.info(f"Added custom prompt: {name}")
    
    def get_available_prompts(self) -> List[str]:
        """Get list of available analysis prompts"""
        return list(self.analysis_prompts.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current vision model"""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "is_initialized": self.is_initialized,
            "available_prompts": self.get_available_prompts()
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("Vision handler cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Test function
def test_vision_handler():
    """Test the vision handler"""
    handler = VisionHandler()
    
    print("Testing Vision handler...")
    
    # Test initialization
    if not handler.initialize():
        print("Failed to initialize Vision handler")
        return False
    
    # Create a test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test basic analysis
    result = handler.analyze_screen(test_image, "general")
    if result:
        print(f"Analysis result: {result}")
    else:
        print("Failed to analyze screen")
        return False
    
    # Test application identification
    app_info = handler.identify_application(test_image)
    if app_info:
        print(f"Application info: {app_info}")
    
    # Test model info
    model_info = handler.get_model_info()
    print(f"Model info: {model_info}")
    
    handler.cleanup()
    print("Vision handler test completed")
    return True

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_vision_handler()
