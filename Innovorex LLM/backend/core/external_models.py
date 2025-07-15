import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from abc import ABC, abstractmethod

# External API clients
import openai
import anthropic
import google.generativeai as genai
import httpx

from ..utils.config import settings

logger = logging.getLogger(__name__)

class ExternalModelInterface(ABC):
    """Abstract base class for external model integrations"""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.is_available = False
        self.last_error = None
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize the model client"""
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        conversation_history: List[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response from the model"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass
    
    def is_configured(self) -> bool:
        """Check if the model is properly configured"""
        return self.api_key is not None and self.api_key.strip() != ""

class OpenAIClient(ExternalModelInterface):
    """OpenAI GPT integration"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.client = None
        super().__init__(api_key, model_name)
    
    def _initialize(self):
        """Initialize OpenAI client"""
        if not self.is_configured():
            logger.warning("OpenAI API key not configured")
            return
        
        try:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            self.is_available = True
            logger.info(f"OpenAI client initialized for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.last_error = str(e)
    
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        conversation_history: List[Dict] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        if not self.is_available:
            raise RuntimeError("OpenAI client not available")
        
        try:
            start_time = time.time()
            
            # Prepare messages
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-10:]:  # Limit history
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Generate response
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            generation_time = time.time() - start_time
            
            # Extract response
            response_text = response.choices[0].message.content
            
            return {
                "response": response_text,
                "generation_time_ms": int(generation_time * 1000),
                "model_name": self.model_name,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason,
                "provider": "openai"
            }
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            self.last_error = str(e)
            raise RuntimeError(f"OpenAI generation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information"""
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "is_available": self.is_available,
            "is_configured": self.is_configured(),
            "context_length": 4096 if "gpt-3.5" in self.model_name else 8192,
            "supports_system_prompt": True,
            "supports_conversation_history": True,
            "last_error": self.last_error
        }

class AnthropicClient(ExternalModelInterface):
    """Anthropic Claude integration"""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-haiku-20240307"):
        self.client = None
        super().__init__(api_key, model_name)
    
    def _initialize(self):
        """Initialize Anthropic client"""
        if not self.is_configured():
            logger.warning("Anthropic API key not configured")
            return
        
        try:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            self.is_available = True
            logger.info(f"Anthropic client initialized for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            self.last_error = str(e)
    
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        conversation_history: List[Dict] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using Anthropic API"""
        if not self.is_available:
            raise RuntimeError("Anthropic client not available")
        
        try:
            start_time = time.time()
            
            # Prepare messages
            messages = []
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-10:]:  # Limit history
                    role = "user" if msg["role"] == "user" else "assistant"
                    messages.append({
                        "role": role,
                        "content": msg["content"]
                    })
            
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Generate response
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "You are a helpful assistant.",
                messages=messages,
                **kwargs
            )
            
            generation_time = time.time() - start_time
            
            # Extract response
            response_text = response.content[0].text
            
            return {
                "response": response_text,
                "generation_time_ms": int(generation_time * 1000),
                "model_name": self.model_name,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "stop_reason": response.stop_reason,
                "provider": "anthropic"
            }
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            self.last_error = str(e)
            raise RuntimeError(f"Anthropic generation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information"""
        context_lengths = {
            "claude-3-haiku-20240307": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-5-sonnet-20240620": 200000
        }
        
        return {
            "provider": "anthropic",
            "model_name": self.model_name,
            "is_available": self.is_available,
            "is_configured": self.is_configured(),
            "context_length": context_lengths.get(self.model_name, 200000),
            "supports_system_prompt": True,
            "supports_conversation_history": True,
            "multimodal": True,
            "last_error": self.last_error
        }

class GoogleClient(ExternalModelInterface):
    """Google Gemini integration"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.client = None
        super().__init__(api_key, model_name)
    
    def _initialize(self):
        """Initialize Google client"""
        if not self.is_configured():
            logger.warning("Google API key not configured")
            return
        
        try:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            self.is_available = True
            logger.info(f"Google client initialized for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Google client: {e}")
            self.last_error = str(e)
    
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        conversation_history: List[Dict] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using Google Gemini API"""
        if not self.is_available:
            raise RuntimeError("Google client not available")
        
        try:
            start_time = time.time()
            
            # Prepare conversation context
            full_prompt = ""
            
            if system_prompt:
                full_prompt += f"Instructions: {system_prompt}\n\n"
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-10:]:  # Limit history
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    full_prompt += f"{role}: {msg['content']}\n"
            
            # Add current prompt
            full_prompt += f"Human: {prompt}\nAssistant:"
            
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
            )
            
            generation_time = time.time() - start_time
            
            # Extract response
            response_text = response.text
            
            # Token usage (approximate)
            input_tokens = len(full_prompt.split()) * 1.3  # Rough estimate
            output_tokens = len(response_text.split()) * 1.3
            
            return {
                "response": response_text,
                "generation_time_ms": int(generation_time * 1000),
                "model_name": self.model_name,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "total_tokens": int(input_tokens + output_tokens),
                "finish_reason": response.candidates[0].finish_reason.name if response.candidates else "unknown",
                "provider": "google"
            }
            
        except Exception as e:
            logger.error(f"Google generation error: {e}")
            self.last_error = str(e)
            raise RuntimeError(f"Google generation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Google model information"""
        context_lengths = {
            "gemini-1.5-flash": 1000000,
            "gemini-1.5-pro": 2000000,
            "gemini-1.0-pro": 32768
        }
        
        return {
            "provider": "google",
            "model_name": self.model_name,
            "is_available": self.is_available,
            "is_configured": self.is_configured(),
            "context_length": context_lengths.get(self.model_name, 32768),
            "supports_system_prompt": True,
            "supports_conversation_history": True,
            "multimodal": True,
            "last_error": self.last_error
        }

class ExternalModelManager:
    """Manager for all external model integrations"""
    
    def __init__(self):
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available external models"""
        # OpenAI models
        if settings.openai_api_key:
            self.models["gpt-3.5-turbo"] = OpenAIClient(
                settings.openai_api_key, 
                "gpt-3.5-turbo"
            )
            self.models["gpt-4"] = OpenAIClient(
                settings.openai_api_key, 
                "gpt-4"
            )
            self.models["gpt-4-turbo"] = OpenAIClient(
                settings.openai_api_key, 
                "gpt-4-turbo"
            )
        
        # Anthropic models
        if settings.anthropic_api_key:
            self.models["claude-3-haiku"] = AnthropicClient(
                settings.anthropic_api_key, 
                "claude-3-haiku-20240307"
            )
            self.models["claude-3-sonnet"] = AnthropicClient(
                settings.anthropic_api_key, 
                "claude-3-sonnet-20240229"
            )
            self.models["claude-3-5-sonnet"] = AnthropicClient(
                settings.anthropic_api_key, 
                "claude-3-5-sonnet-20240620"
            )
        
        # Google models
        if settings.google_api_key:
            self.models["gemini-1.5-flash"] = GoogleClient(
                settings.google_api_key, 
                "gemini-1.5-flash"
            )
            self.models["gemini-1.5-pro"] = GoogleClient(
                settings.google_api_key, 
                "gemini-1.5-pro"
            )
            self.models["gemini-1.0-pro"] = GoogleClient(
                settings.google_api_key, 
                "gemini-1.0-pro"
            )
    
    def get_available_models(self) -> List[str]:
        """Get list of available external models"""
        return [
            model_name for model_name, client in self.models.items()
            if client.is_available
        ]
    
    def get_model_client(self, model_name: str) -> Optional[ExternalModelInterface]:
        """Get model client by name"""
        return self.models.get(model_name)
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available"""
        client = self.models.get(model_name)
        return client is not None and client.is_available
    
    async def generate_response(
        self,
        model_name: str,
        prompt: str,
        system_prompt: str = None,
        conversation_history: List[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using specified external model"""
        client = self.get_model_client(model_name)
        
        if not client:
            raise ValueError(f"Model {model_name} not found")
        
        if not client.is_available:
            raise RuntimeError(f"Model {model_name} not available: {client.last_error}")
        
        return await client.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            **kwargs
        )
    
    def get_all_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all models"""
        return {
            model_name: client.get_model_info()
            for model_name, client in self.models.items()
        }
    
    async def compare_models(
        self,
        model_names: List[str],
        prompt: str,
        system_prompt: str = None,
        conversation_history: List[Dict] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Compare responses from multiple models"""
        results = {}
        
        # Generate responses from all models concurrently
        tasks = []
        for model_name in model_names:
            if self.is_model_available(model_name):
                task = self.generate_response(
                    model_name=model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    conversation_history=conversation_history,
                    **kwargs
                )
                tasks.append((model_name, task))
        
        # Wait for all responses
        for model_name, task in tasks:
            try:
                result = await task
                results[model_name] = result
            except Exception as e:
                logger.error(f"Error generating response from {model_name}: {e}")
                results[model_name] = {
                    "error": str(e),
                    "model_name": model_name
                }
        
        return results
    
    def refresh_models(self):
        """Refresh all model connections"""
        logger.info("Refreshing external model connections...")
        self.models.clear()
        self._initialize_models()
        
        available_count = len(self.get_available_models())
        logger.info(f"Refreshed external models: {available_count} available")

# Global external model manager
external_model_manager = ExternalModelManager()