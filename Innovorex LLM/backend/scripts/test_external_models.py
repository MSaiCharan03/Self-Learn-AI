#!/usr/bin/env python3
"""
Test script for external model integrations
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from core.external_models import external_model_manager

async def test_external_models():
    """Test external model integrations"""
    print("Testing External Model Integrations")
    print("=" * 60)
    
    # Get available models
    available_models = external_model_manager.get_available_models()
    all_models = external_model_manager.get_all_model_info()
    
    print(f"Configured models: {len(all_models)}")
    print(f"Available models: {len(available_models)}")
    
    if not available_models:
        print("\n❌ No external models are available.")
        print("Make sure to configure API keys in your .env file:")
        print("  OPENAI_API_KEY=your_openai_key")
        print("  ANTHROPIC_API_KEY=your_anthropic_key") 
        print("  GOOGLE_API_KEY=your_google_key")
        return
    
    # Display model information
    print(f"\nAvailable models: {', '.join(available_models)}")
    
    print("\nModel Information:")
    for model_name, info in all_models.items():
        status = "✅ Available" if info["is_available"] else "❌ Unavailable"
        print(f"  {model_name} ({info['provider']}): {status}")
        
        if info["is_available"]:
            print(f"    Context length: {info['context_length']:,}")
            print(f"    Multimodal: {info.get('multimodal', False)}")
        elif info.get("last_error"):
            print(f"    Error: {info['last_error']}")
    
    # Test generation with each available model
    test_prompt = "What are the key benefits of renewable energy?"
    system_prompt = "You are a helpful assistant. Provide a concise but informative answer."
    
    print(f"\n\nTesting generation with prompt: '{test_prompt}'")
    print("=" * 60)
    
    for model_name in available_models:
        print(f"\n{model_name}:")
        print("-" * 40)
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            response = await external_model_manager.generate_response(
                model_name=model_name,
                prompt=test_prompt,
                system_prompt=system_prompt,
                max_tokens=200,
                temperature=0.7
            )
            
            print(f"Response: {response['response']}")
            print(f"Generation time: {response['generation_time_ms']}ms")
            print(f"Tokens: {response['input_tokens']} in → {response['output_tokens']} out")
            print(f"Provider: {response['provider']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test model comparison
    if len(available_models) > 1:
        print(f"\n\nTesting model comparison:")
        print("=" * 60)
        
        try:
            comparison_prompt = "Explain machine learning in simple terms."
            
            results = await external_model_manager.compare_models(
                model_names=available_models[:3],  # Test up to 3 models
                prompt=comparison_prompt,
                system_prompt=system_prompt,
                max_tokens=150,
                temperature=0.7
            )
            
            for model_name, result in results.items():
                print(f"\n{model_name}:")
                if "error" in result:
                    print(f"  ❌ Error: {result['error']}")
                else:
                    print(f"  Response: {result['response'][:100]}...")
                    print(f"  Time: {result['generation_time_ms']}ms")
                    print(f"  Tokens: {result['output_tokens']}")
            
        except Exception as e:
            print(f"❌ Comparison error: {e}")
    
    print(f"\n\n✅ External model testing completed!")

async def test_conversation_history():
    """Test conversation history handling"""
    print("\n\nTesting Conversation History")
    print("=" * 60)
    
    available_models = external_model_manager.get_available_models()
    
    if not available_models:
        print("No models available for testing")
        return
    
    # Test with conversation history
    conversation_history = [
        {"role": "user", "content": "Hello, what's the weather like?"},
        {"role": "assistant", "content": "I don't have access to current weather data, but I can help you with other questions!"},
        {"role": "user", "content": "That's okay. Can you tell me about Python programming?"}
    ]
    
    test_model = available_models[0]
    
    try:
        print(f"Testing {test_model} with conversation history...")
        
        response = await external_model_manager.generate_response(
            model_name=test_model,
            prompt="What are some good Python libraries for data science?",
            system_prompt="You are a helpful programming assistant.",
            conversation_history=conversation_history,
            max_tokens=200,
            temperature=0.7
        )
        
        print(f"Response: {response['response']}")
        print(f"Generation time: {response['generation_time_ms']}ms")
        
    except Exception as e:
        print(f"❌ Error testing conversation history: {e}")

def main():
    """Main test function"""
    print("External Model Integration Tests")
    print("=" * 60)
    
    asyncio.run(test_external_models())
    asyncio.run(test_conversation_history())

if __name__ == "__main__":
    main()