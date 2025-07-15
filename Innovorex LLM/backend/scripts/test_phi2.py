#!/usr/bin/env python3
"""
Test script for Phi-2 model integration
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import time
from core.llm_manager import phi2_manager

async def test_phi2_basic():
    """Test basic Phi-2 functionality"""
    print("Testing Phi-2 Basic Functionality")
    print("=" * 50)
    
    # Test model info
    print("1. Model Info:")
    info = phi2_manager.get_model_info()
    print(f"   Loaded: {info['is_loaded']}")
    
    if info['is_loaded']:
        print(f"   Model: {info['model_info']['model_name']}")
        print(f"   Vocab Size: {info['model_info']['vocab_size']}")
        print(f"   Context Length: {info['model_info']['context_length']}")
        print(f"   Device: {info['device']}")
        print(f"   Memory: {info['memory_usage']}")
    
    if not info['is_loaded']:
        print("   Model not loaded - cannot proceed with tests")
        return
    
    # Test simple generation
    print("\n2. Simple Generation:")
    test_prompt = "What is artificial intelligence?"
    
    start_time = time.time()
    try:
        response = await phi2_manager.generate_response_async(
            prompt=test_prompt,
            generation_params={
                "max_new_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        )
        
        print(f"   Prompt: {test_prompt}")
        print(f"   Response: {response['response']}")
        print(f"   Generation time: {response['generation_time_ms']}ms")
        print(f"   Input tokens: {response['input_tokens']}")
        print(f"   Output tokens: {response['output_tokens']}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test with conversation history
    print("\n3. Conversation History Test:")
    conversation_history = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": "What can you help me with?"}
    ]
    
    try:
        response = await phi2_manager.generate_response_async(
            prompt="Tell me about machine learning",
            conversation_history=conversation_history,
            generation_params={
                "max_new_tokens": 150,
                "temperature": 0.8
            }
        )
        
        print(f"   Response: {response['response']}")
        print(f"   Generation time: {response['generation_time_ms']}ms")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test with system prompt
    print("\n4. System Prompt Test:")
    system_prompt = "You are a helpful coding assistant. Provide concise, practical answers about programming."
    
    try:
        response = await phi2_manager.generate_response_async(
            prompt="How do I create a Python function?",
            system_prompt=system_prompt,
            generation_params={
                "max_new_tokens": 200,
                "temperature": 0.6
            }
        )
        
        print(f"   System prompt: {system_prompt}")
        print(f"   Response: {response['response']}")
        print(f"   Generation time: {response['generation_time_ms']}ms")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test tokenization
    print("\n5. Tokenization Test:")
    test_text = "This is a test sentence for tokenization."
    
    try:
        tokens = phi2_manager.tokenize(test_text)
        print(f"   Text: {test_text}")
        print(f"   Token count: {tokens['token_count']}")
        print(f"   Tokens: {tokens['tokens'][:10]}...")  # Show first 10 tokens
        
    except Exception as e:
        print(f"   Error: {e}")

async def test_phi2_performance():
    """Test Phi-2 performance with multiple requests"""
    print("\n\nTesting Phi-2 Performance")
    print("=" * 50)
    
    if not phi2_manager.is_loaded:
        print("Model not loaded - cannot test performance")
        return
    
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "What are the benefits of renewable energy?",
        "How does photosynthesis work?",
        "Describe the process of machine learning.",
        "What is the importance of biodiversity?"
    ]
    
    total_time = 0
    total_tokens = 0
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Testing prompt: {prompt[:50]}...")
        
        try:
            start_time = time.time()
            response = await phi2_manager.generate_response_async(
                prompt=prompt,
                generation_params={
                    "max_new_tokens": 100,
                    "temperature": 0.7
                }
            )
            
            generation_time = response['generation_time_ms']
            output_tokens = response['output_tokens']
            
            total_time += generation_time
            total_tokens += output_tokens
            
            print(f"   Time: {generation_time}ms")
            print(f"   Tokens: {output_tokens}")
            print(f"   Tokens/sec: {output_tokens / (generation_time / 1000):.1f}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    if total_time > 0:
        print(f"\nPerformance Summary:")
        print(f"   Total time: {total_time}ms")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Average time per request: {total_time / len(test_prompts):.1f}ms")
        print(f"   Average tokens per request: {total_tokens / len(test_prompts):.1f}")
        print(f"   Overall tokens/sec: {total_tokens / (total_time / 1000):.1f}")

async def test_phi2_error_handling():
    """Test error handling"""
    print("\n\nTesting Error Handling")
    print("=" * 50)
    
    if not phi2_manager.is_loaded:
        print("Model not loaded - cannot test error handling")
        return
    
    # Test with very long prompt
    print("1. Long prompt test:")
    long_prompt = "This is a very long prompt. " * 500  # ~3000 tokens
    
    try:
        response = await phi2_manager.generate_response_async(
            prompt=long_prompt,
            generation_params={"max_new_tokens": 50}
        )
        print(f"   Response length: {len(response['response'])}")
        print(f"   Generation time: {response['generation_time_ms']}ms")
        
    except Exception as e:
        print(f"   Expected error: {e}")
    
    # Test with empty prompt
    print("\n2. Empty prompt test:")
    try:
        response = await phi2_manager.generate_response_async(
            prompt="",
            generation_params={"max_new_tokens": 50}
        )
        print(f"   Response: {response['response']}")
        
    except Exception as e:
        print(f"   Expected error: {e}")

async def main():
    """Run all tests"""
    print("Phi-2 Model Integration Tests")
    print("=" * 60)
    
    await test_phi2_basic()
    await test_phi2_performance()
    await test_phi2_error_handling()
    
    print("\n" + "=" * 60)
    print("Tests completed!")

if __name__ == "__main__":
    asyncio.run(main())