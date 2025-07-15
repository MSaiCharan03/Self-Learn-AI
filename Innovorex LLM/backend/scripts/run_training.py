#!/usr/bin/env python3
"""
Script to run model fine-tuning based on user feedback
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import argparse
from sqlalchemy.orm import Session
from core.database import SessionLocal
from core.fine_tuning import fine_tuning_manager
from models.feedback import TrainingSession

def run_training(
    training_type: str = "lora",
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    rank: int = 8,
    alpha: int = 32,
    min_examples: int = 10
):
    """Run fine-tuning training"""
    print(f"Starting {training_type} training...")
    print("=" * 50)
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Check available training data
        print("1. Checking available training data...")
        
        # Get data statistics
        from core.fine_tuning import FineTuningDataProcessor
        from transformers import AutoTokenizer
        from utils.config import settings
        
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            cache_dir=settings.phi2_model_path,
            trust_remote_code=True
        )
        
        data_processor = FineTuningDataProcessor(tokenizer)
        
        # Extract data
        conversation_pairs = data_processor.extract_conversation_pairs(db)
        preference_pairs = data_processor.extract_preference_pairs(db)
        negative_examples = data_processor.extract_negative_examples(db)
        
        print(f"   Conversation pairs: {len(conversation_pairs)}")
        print(f"   Preference pairs: {len(preference_pairs)}")
        print(f"   Negative examples: {len(negative_examples)}")
        
        if len(conversation_pairs) < min_examples:
            print(f"❌ Not enough training examples. Need at least {min_examples}, got {len(conversation_pairs)}")
            print("   Please collect more user feedback before training.")
            return
        
        # Display sample data
        if conversation_pairs:
            print("\n   Sample conversation pair:")
            sample = conversation_pairs[0]
            print(f"   Input: {sample['input'][:100]}...")
            print(f"   Output: {sample['output'][:100]}...")
            print(f"   Rating: {sample['rating']}")
        
        # Start training session
        print("\n2. Starting training session...")
        
        session = fine_tuning_manager.start_training_session(
            db=db,
            training_type=training_type,
            parameters={
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "rank": rank,
                "alpha": alpha
            }
        )
        
        print(f"   Training session ID: {session.id}")
        print(f"   Training type: {session.training_type}")
        print(f"   Status: {session.status}")
        
        # Run training
        print("\n3. Running training...")
        
        if training_type == "lora":
            results = fine_tuning_manager.run_lora_training(
                db=db,
                session=session,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                rank=rank,
                alpha=alpha
            )
        else:
            print(f"❌ Training type '{training_type}' not supported yet")
            return
        
        # Display results
        print("\n4. Training Results:")
        
        if results["success"]:
            print("✅ Training completed successfully!")
            print(f"   Session ID: {results['session_id']}")
            print(f"   Output directory: {results['output_dir']}")
            print(f"   Training examples: {results['data_stats']['dataset_size']}")
            print(f"   Average rating: {results['data_stats']['average_rating']:.2f}")
            
            training_results = results["training_results"]
            print(f"   Final loss: {training_results['train_loss']:.4f}")
            print(f"   Training time: {training_results['training_time']:.2f} seconds")
            print(f"   Samples per second: {training_results['train_samples_per_second']:.2f}")
            
        else:
            print("❌ Training failed!")
            print(f"   Error: {results['error']}")
            print(f"   Session ID: {results['session_id']}")
        
        # Evaluate improvement
        if results["success"]:
            print("\n5. Evaluating model improvement...")
            
            try:
                evaluation_results = fine_tuning_manager.evaluate_model_improvement(
                    db=db,
                    checkpoint_path=results["output_dir"]
                )
                
                if "error" not in evaluation_results:
                    print("✅ Model evaluation completed")
                    print(f"   Original model responses: {len(evaluation_results['original_model'])}")
                    
                    # Show sample comparison
                    if evaluation_results["original_model"]:
                        sample = evaluation_results["original_model"][0]
                        print(f"\n   Sample comparison:")
                        print(f"   Prompt: {sample['prompt']}")
                        print(f"   Original response: {sample['response'][:200]}...")
                        print(f"   Generation time: {sample['generation_time']}ms")
                        
                else:
                    print(f"❌ Model evaluation failed: {evaluation_results['error']}")
                    
            except Exception as e:
                print(f"❌ Model evaluation error: {e}")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        
        # Mark session as failed if it exists
        try:
            if 'session' in locals():
                session.status = "failed"
                session.error_log = str(e)
                db.commit()
        except:
            pass
    
    finally:
        db.close()

def list_training_sessions():
    """List all training sessions"""
    print("Training Sessions History")
    print("=" * 50)
    
    db = SessionLocal()
    
    try:
        sessions = db.query(TrainingSession).order_by(
            TrainingSession.start_time.desc()
        ).limit(10).all()
        
        if not sessions:
            print("No training sessions found.")
            return
        
        for session in sessions:
            print(f"Session {session.id}:")
            print(f"  Model: {session.model_name}")
            print(f"  Type: {session.training_type}")
            print(f"  Status: {session.status}")
            print(f"  Started: {session.start_time}")
            print(f"  Data size: {session.data_size}")
            
            if session.end_time:
                duration = session.end_time - session.start_time
                print(f"  Duration: {duration}")
            
            if session.error_log:
                print(f"  Error: {session.error_log[:100]}...")
            
            print()
        
    except Exception as e:
        print(f"Error listing sessions: {e}")
    
    finally:
        db.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run model fine-tuning")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Start training")
    train_parser.add_argument("--type", default="lora", choices=["lora"], help="Training type")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    train_parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    train_parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    train_parser.add_argument("--min-examples", type=int, default=10, help="Minimum training examples")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List training sessions")
    
    args = parser.parse_args()
    
    if args.command == "train":
        run_training(
            training_type=args.type,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            rank=args.rank,
            alpha=args.alpha,
            min_examples=args.min_examples
        )
    elif args.command == "list":
        list_training_sessions()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()