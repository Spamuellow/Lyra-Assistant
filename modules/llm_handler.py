"""
LLM Handler
Handles language model interactions using Ollama for local inference
"""

import requests
import json
import logging
import time
from typing import Optional, Dict, Any, List, Generator
from pathlib import Path
import threading
import queue

from config import MODELS, ASSISTANT, DEBUG

logger = logging.getLogger(__name__)

class LLMHandler:
    """Handles language model operations using Ollama"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = None
        self.is_initialized = False
        
        # Conversation context
        self.context = []
        self.max_context_length = ASSISTANT["context_window"]
        self.memory_length = ASSISTANT["memory_length"]
        
        # Model settings
        self.max_tokens = MODELS["llm"]["max_tokens"]
        self.temperature = MODELS["llm"]["temperature"]
        self.top_p = MODELS["llm"]["top_p"]
        
        # Assistant personality
        self.assistant_name = ASSISTANT["name"]
        self.personality = ASSISTANT["personality"]
        self.roleplay_mode = ASSISTANT["roleplay_mode"]
        self.uncensored = ASSISTANT["uncensored"]
        
        # System prompt
        self.system_prompt = self._build_system_prompt()
        
        logger.info("LLMHandler initialized")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the assistant"""
        prompt = f"""You are {self.assistant_name}, a helpful AI assistant with these characteristics:
- Personality: {self.personality}
- You can see the user's screen and comment on what they're doing
- You have access to real-time screen capture data
- You can engage in conversations about games, applications, and screen content
- You maintain context from previous interactions
"""
        
        if self.roleplay_mode:
            prompt += """
- You excel at roleplay and creative interactions
- You can adapt your personality and speaking style as requested
- You're comfortable with creative and imaginative scenarios
"""
        
        if self.uncensored:
            prompt += """
- You can discuss mature topics openly when appropriate
- You don't have unnecessary content restrictions for roleplay
- You focus on being helpful rather than overly cautious
"""
        
        prompt += """
- Always be engaging and conversational
- Keep responses natural and not overly formal
- Show interest in what the user is doing on their screen
- Ask follow-up questions to keep conversations flowing
- Be concise but informative
"""
        
        return prompt
    
    def initialize(self, model_name: str = None) -> bool:
        """Initialize the LLM handler with a specific model"""
        try:
            logger.info("Initializing LLM handler...")
            
            # Check if Ollama is running
            if not self._check_ollama_status():
                logger.error("Ollama is not running. Please start Ollama first.")
                return False
            
            # Set model name
            if model_name:
                self.model_name = model_name
            else:
                # Auto-detect available models
                available_models = self._get_available_models()
                if not available_models:
                    logger.error("No models available in Ollama")
                    return False
                
                # Select first available model (or implement selection logic)
                self.model_name = available_models[0]
            
            logger.info(f"Using model: {self.model_name}")
            
            # Test model with a simple prompt
            test_response = self._generate_response("Hello", test_mode=True)
            if test_response:
                logger.info("LLM model test successful")
                self.is_initialized = True
                return True
            else:
                logger.error("LLM model test failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM handler: {e}")
            return False
    
    def _check_ollama_status(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                logger.info(f"Available models: {models}")
                return models
            return []
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def generate_response(self, prompt: str, screen_context: str = None) -> Optional[str]:
        """Generate a response to a prompt with optional screen context"""
        if not self.is_initialized:
            logger.error("LLM handler not initialized")
            return None
        
        try:
            # Build full prompt with context
            full_prompt = self._build_full_prompt(prompt, screen_context)
            
            # Generate response
            response = self._generate_response(full_prompt)
            
            if response:
                # Add to context
                self._add_to_context(prompt, response, screen_context)
                
                logger.info(f"Generated response: {response[:100]}{'...' if len(response) > 100 else ''}")
                return response
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return None
    
    def _build_full_prompt(self, user_prompt: str, screen_context: str = None) -> str:
        """Build the full prompt including system message and context"""
        # Start with system prompt
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history
        for ctx in self.context[-self.memory_length:]:
            messages.append({"role": "user", "content": ctx["user"]})
            messages.append({"role": "assistant", "content": ctx["assistant"]})
        
        # Add current prompt with screen context
        current_prompt = user_prompt
        if screen_context:
            current_prompt = f"[Screen Context: {screen_context}]\n\nUser: {user_prompt}"
        
        messages.append({"role": "user", "content": current_prompt})
        
        # Convert to string format for Ollama
        full_prompt = ""
        for message in messages:
            if message["role"] == "system":
                full_prompt += f"System: {message['content']}\n\n"
            elif message["role"] == "user":
                full_prompt += f"User: {message['content']}\n\n"
            elif message["role"] == "assistant":
                full_prompt += f"Assistant: {message['content']}\n\n"
        
        full_prompt += "Assistant: "
        
        return full_prompt
    
    def _generate_response(self, prompt: str, test_mode: bool = False) -> Optional[str]:
        """Generate response using Ollama API"""
        try:
            start_time = time.time()
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                },
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data.get("response", "").strip()
                
                if DEBUG["log_timings"]:
                    generation_time = time.time() - start_time
                    logger.debug(f"LLM generation took {generation_time:.2f}s")
                
                return generated_text
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return None
    
    def generate_stream(self, prompt: str, screen_context: str = None) -> Generator[str, None, None]:
        """Generate streaming response"""
        if not self.is_initialized:
            logger.error("LLM handler not initialized")
            return
        
        try:
            full_prompt = self._build_full_prompt(prompt, screen_context)
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                },
                "stream": True
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            token = data["response"]
                            full_response += token
                            yield token
                        
                        if data.get("done", False):
                            break
                
                # Add to context
                self._add_to_context(prompt, full_response, screen_context)
                
        except Exception as e:
            logger.error(f"Failed to generate streaming response: {e}")
    
    def _add_to_context(self, user_prompt: str, assistant_response: str, screen_context: str = None):
        """Add interaction to context"""
        context_entry = {
            "user": user_prompt,
            "assistant": assistant_response,
            "screen_context": screen_context,
            "timestamp": time.time()
        }
        
        self.context.append(context_entry)
        
        # Trim context if too long
        if len(self.context) > self.memory_length:
            self.context = self.context[-self.memory_length:]
    
    def clear_context(self):
        """Clear conversation context"""
        self.context = []
        logger.info("Context cleared")
    
    def get_context_summary(self) -> str:
        """Get a summary of recent context"""
        if not self.context:
            return "No recent context"
        
        summary = f"Recent conversation ({len(self.context)} exchanges):\n"
        for i, ctx in enumerate(self.context[-5:], 1):
            summary += f"{i}. User: {ctx['user'][:50]}...\n"
            summary += f"   Assistant: {ctx['assistant'][:50]}...\n"
        
        return summary
    
    def analyze_screen_content(self, screen_description: str) -> Optional[str]:
        """Analyze screen content and provide commentary"""
        if not self.is_initialized:
            return None
        
        analysis_prompt = f"""Analyze this screen content and provide a brief, engaging commentary:

Screen Description: {screen_description}

Please:
1. Identify what the user is doing
2. Make relevant observations or comments
3. Ask an engaging follow-up question if appropriate
4. Keep it conversational and natural

Your response:"""
        
        return self._generate_response(analysis_prompt)
    
    def set_personality(self, personality: str):
        """Update the assistant's personality"""
        self.personality = personality
        self.system_prompt = self._build_system_prompt()
        logger.info(f"Personality updated to: {personality}")
    
    def set_roleplay_character(self, character_description: str):
        """Set a specific roleplay character"""
        roleplay_prompt = f"""You are now roleplaying as: {character_description}

Stay in character while maintaining your helpful nature. You can still:
- Comment on screen content
- Help with tasks
- Engage in conversations

But respond as this character would, with their personality, mannerisms, and perspective.
"""
        
        self.system_prompt = roleplay_prompt
        logger.info(f"Roleplay character set: {character_description}")
    
    def reset_to_default(self):
        """Reset to default personality"""
        self.system_prompt = self._build_system_prompt()
        logger.info("Reset to default personality")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "context_length": len(self.context),
            "is_initialized": self.is_initialized
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.context = []
            logger.info("LLM handler cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Test function
def test_llm_handler():
    """Test the LLM handler"""
    handler = LLMHandler()
    
    print("Testing LLM handler...")
    
    # Test initialization
    if not handler.initialize():
        print("Failed to initialize LLM handler")
        return False
    
    # Test basic response
    response = handler.generate_response("Hello, how are you?")
    if response:
        print(f"Response: {response}")
    else:
        print("Failed to generate response")
        return False
    
    # Test with screen context
    screen_context = "User is browsing a website about AI development"
    response = handler.generate_response("What do you think about this?", screen_context)
    if response:
        print(f"Response with context: {response}")
    
    # Test context
    print(f"Context summary: {handler.get_context_summary()}")
    
    handler.cleanup()
    print("LLM handler test completed")
    return True

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_llm_handler()
