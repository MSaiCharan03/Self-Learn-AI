import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextStreamer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, AsyncGenerator
from threading import Lock
from datetime import datetime
from ..utils.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for generation"""
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class Phi2Manager:
    """Manages the local Phi-2 model for CPU inference"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = settings.phi2_model_path
        self.device = "cpu"  # Force CPU usage
        self.is_loaded = False
        self.load_lock = Lock()
        self.generation_lock = Lock()
        self.model_info = {}
        
        # Generation parameters
        self.default_generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "do_sample": True,
            "pad_token_id": None,  # Will be set after tokenizer loads
            "eos_token_id": None,  # Will be set after tokenizer loads
            "repetition_penalty": 1.1,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
        }
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self):
        """Load the Phi-2 model and tokenizer"""
        with self.load_lock:
            if self.is_loaded:
                return
            
            try:
                logger.info("Loading Phi-2 model...")
                start_time = time.time()
                
                # Create model directory if it doesn't exist
                os.makedirs(self.model_path, exist_ok=True)
                
                # Load tokenizer
                logger.info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/phi-2",
                    cache_dir=self.model_path,
                    trust_remote_code=True,
                    local_files_only=False
                )
                
                # Set pad token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model
                logger.info("Loading model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/phi-2",
                    cache_dir=self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    device_map="cpu",
                    local_files_only=False
                )
                
                # Set model to evaluation mode
                self.model.eval()
                
                # Update generation config with tokenizer info
                self.default_generation_config["pad_token_id"] = self.tokenizer.pad_token_id
                self.default_generation_config["eos_token_id"] = self.tokenizer.eos_token_id
                
                # Store model info
                self.model_info = {
                    "model_name": "microsoft/phi-2",
                    "model_type": "phi",
                    "device": self.device,
                    "vocab_size": self.tokenizer.vocab_size,
                    "context_length": 2048,
                    "loaded_at": datetime.now().isoformat(),
                    "load_time_seconds": time.time() - start_time
                }
                
                self.is_loaded = True
                logger.info(f"Phi-2 model loaded successfully in {self.model_info['load_time_seconds']:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Failed to load Phi-2 model: {e}")
                raise RuntimeError(f"Model loading failed: {e}")
    
    def _prepare_prompt(self, user_input: str, system_prompt: str = None, conversation_history: List[Dict] = None) -> str:
        """Prepare the prompt for Phi-2 generation"""
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = (
                "You are a helpful, harmless, and honest AI assistant. "
                "Provide clear, accurate, and helpful responses to the user's questions. "
                "Be concise but informative."
            )
        
        # Format conversation history if provided
        context = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Keep last 5 messages for context
                if msg["role"] == "user":
                    context += f"Human: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    context += f"Assistant: {msg['content']}\n"
        
        # Construct final prompt
        prompt = f"{system_prompt}\n\n{context}Human: {user_input}\nAssistant:"
        
        return prompt
    
    def _post_process_response(self, generated_text: str, prompt: str) -> str:
        """Post-process the generated response"""
        try:
            # Remove the original prompt from the response
            if prompt in generated_text:
                response = generated_text.replace(prompt, "").strip()
            else:
                response = generated_text.strip()
            
            # Remove any trailing incomplete sentences
            if response.endswith("Human:"):
                response = response[:-6].strip()
            
            # Clean up common issues
            response = response.replace("Assistant:", "").strip()
            
            # Remove any repeated patterns
            lines = response.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and line not in cleaned_lines[-2:]:  # Avoid immediate repetition
                    cleaned_lines.append(line)
            
            response = '\n'.join(cleaned_lines)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error post-processing response: {e}")
            return generated_text.strip()
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: str = None,
        conversation_history: List[Dict] = None,
        generation_params: Dict = None
    ) -> Dict:
        """Generate a response using Phi-2"""
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Please ensure the model is properly initialized.")
        
        with self.generation_lock:
            try:
                start_time = time.time()
                
                # Prepare the full prompt
                full_prompt = self._prepare_prompt(prompt, system_prompt, conversation_history)
                
                # Tokenize input
                inputs = self.tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1800,  # Leave room for generation
                    padding=True
                ).to(self.device)
                
                # Merge generation parameters
                gen_config = self.default_generation_config.copy()
                if generation_params:
                    gen_config.update(generation_params)
                
                # Setup stopping criteria
                stop_tokens = [self.tokenizer.eos_token_id]
                if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                    # Add common stop patterns
                    stop_patterns = ["Human:", "\nHuman:", "User:", "\nUser:"]
                    for pattern in stop_patterns:
                        try:
                            token_ids = self.tokenizer.encode(pattern, add_special_tokens=False)
                            if token_ids:
                                stop_tokens.extend(token_ids)
                        except:
                            pass
                
                stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_tokens)])
                
                # Generate response
                with torch.inference_mode():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        stopping_criteria=stopping_criteria,
                        **gen_config
                    )
                
                # Decode response
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Post-process response
                response_text = self._post_process_response(generated_text, full_prompt)
                
                # Calculate metrics
                generation_time = time.time() - start_time
                input_tokens = len(inputs.input_ids[0])
                output_tokens = len(outputs[0]) - input_tokens
                
                # Return structured response
                return {
                    "response": response_text,
                    "generation_time_ms": int(generation_time * 1000),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "model_name": "phi-2",
                    "parameters": gen_config,
                    "prompt_length": len(full_prompt),
                    "response_length": len(response_text)
                }
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise RuntimeError(f"Text generation failed: {e}")
    
    async def generate_response_async(
        self,
        prompt: str,
        system_prompt: str = None,
        conversation_history: List[Dict] = None,
        generation_params: Dict = None
    ) -> Dict:
        """Async wrapper for generate_response"""
        import asyncio
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate_response,
            prompt,
            system_prompt,
            conversation_history,
            generation_params
        )
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = None,
        conversation_history: List[Dict] = None,
        generation_params: Dict = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response (placeholder for future implementation)"""
        # For now, return the full response as a single chunk
        # TODO: Implement actual streaming generation
        response = self.generate_response(prompt, system_prompt, conversation_history, generation_params)
        yield response["response"]
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "is_loaded": self.is_loaded,
            "model_info": self.model_info,
            "device": self.device,
            "memory_usage": self._get_memory_usage(),
            "generation_config": self.default_generation_config
        }
    
    def _get_memory_usage(self) -> Dict:
        """Get memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "ram_usage_mb": memory_info.rss / (1024 * 1024),
                "ram_usage_gb": memory_info.rss / (1024 * 1024 * 1024),
                "cpu_percent": process.cpu_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def update_generation_config(self, new_config: Dict):
        """Update generation configuration"""
        self.default_generation_config.update(new_config)
        logger.info(f"Updated generation config: {new_config}")
    
    def tokenize(self, text: str) -> Dict:
        """Tokenize text and return token information"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        tokens = self.tokenizer(text, return_tensors="pt")
        
        return {
            "input_ids": tokens.input_ids.tolist(),
            "attention_mask": tokens.attention_mask.tolist(),
            "token_count": len(tokens.input_ids[0]),
            "tokens": self.tokenizer.convert_ids_to_tokens(tokens.input_ids[0])
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        if not self.is_loaded:
            # Rough estimation: ~4 characters per token
            return len(text) // 4
        
        return len(self.tokenizer.encode(text))
    
    def reload_model(self):
        """Reload the model (useful for updates)"""
        logger.info("Reloading Phi-2 model...")
        self.is_loaded = False
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def unload_model(self):
        """Unload the model to free memory"""
        logger.info("Unloading Phi-2 model...")
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Global model manager instance
phi2_manager = Phi2Manager()