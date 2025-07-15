#!/usr/bin/env python3
"""
Setup script to download and cache the Phi-2 model locally
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.config import settings
import time

def download_phi2_model():
    """Download and cache Phi-2 model locally"""
    print("Starting Phi-2 model download and setup...")
    
    # Create model directory
    model_path = settings.phi2_model_path
    os.makedirs(model_path, exist_ok=True)
    
    print(f"Model will be cached in: {model_path}")
    print(f"Using device: CPU (as configured)")
    
    try:
        # Download tokenizer
        print("\n1. Downloading tokenizer...")
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            cache_dir=model_path,
            trust_remote_code=True
        )
        
        tokenizer_time = time.time() - start_time
        print(f"   Tokenizer downloaded in {tokenizer_time:.2f} seconds")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        
        # Download model
        print("\n2. Downloading model...")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            cache_dir=model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu"
        )
        
        model_time = time.time() - start_time
        print(f"   Model downloaded in {model_time:.2f} seconds")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test generation
        print("\n3. Testing model generation...")
        start_time = time.time()
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test prompt
        test_prompt = "Hello, I am an AI assistant."
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        model.eval()
        with torch.inference_mode():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        test_time = time.time() - start_time
        
        print(f"   Test generation completed in {test_time:.2f} seconds")
        print(f"   Test output: {generated_text}")
        
        # Save model info
        model_info = {
            "model_name": "microsoft/phi-2",
            "model_path": model_path,
            "vocab_size": tokenizer.vocab_size,
            "context_length": 2048,
            "download_time": tokenizer_time + model_time,
            "test_time": test_time,
            "device": "cpu",
            "torch_dtype": "float32"
        }
        
        import json
        with open(os.path.join(model_path, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"\n✅ Model setup completed successfully!")
        print(f"   Total time: {tokenizer_time + model_time + test_time:.2f} seconds")
        print(f"   Model info saved to: {os.path.join(model_path, 'model_info.json')}")
        
        # Memory usage info
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            print(f"   Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            print("   Install psutil to see memory usage")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during model setup: {e}")
        print("   Please check your internet connection and try again.")
        return False

def verify_model_setup():
    """Verify that the model is properly set up"""
    print("\nVerifying model setup...")
    
    model_path = settings.phi2_model_path
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        return False
    
    # Check for model files
    expected_files = ["config.json", "model_info.json"]
    missing_files = []
    
    for file in expected_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        return False
    
    # Try to load model info
    try:
        import json
        with open(os.path.join(model_path, "model_info.json"), "r") as f:
            model_info = json.load(f)
        
        print("✅ Model verification successful!")
        print(f"   Model: {model_info['model_name']}")
        print(f"   Vocab size: {model_info['vocab_size']}")
        print(f"   Context length: {model_info['context_length']}")
        print(f"   Device: {model_info['device']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading model info: {e}")
        return False

def main():
    """Main setup function"""
    print("Phi-2 Model Setup")
    print("=" * 50)
    
    # Check if model is already set up
    if verify_model_setup():
        print("\nModel is already set up!")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Download and setup model
    success = download_phi2_model()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("You can now run the LLM platform with local Phi-2 model.")
    else:
        print("\n💥 Setup failed!")
        print("Please check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()