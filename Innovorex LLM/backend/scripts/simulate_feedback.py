#!/usr/bin/env python3
"""
Script to simulate user feedback for testing the fine-tuning system
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import random
import time
from sqlalchemy.orm import Session
from core.database import SessionLocal
from models.user import User
from models.conversation import Conversation, Message, ModelResponse
from models.feedback import Feedback
from core.llm_manager import phi2_manager
from utils.helpers import create_embedding_for_message, create_embedding_for_response

# Sample prompts for testing
TEST_PROMPTS = [
    "What is artificial intelligence?",
    "How can I improve my productivity?",
    "Explain machine learning in simple terms.",
    "What are the benefits of renewable energy?",
    "How do I learn a new programming language?",
    "What is the difference between AI and machine learning?",
    "How does blockchain technology work?",
    "What are the best practices for software development?",
    "How can I start a career in data science?",
    "What is cloud computing and its advantages?",
    "How do I create a healthy work-life balance?",
    "What are the latest trends in technology?",
    "How do I improve my problem-solving skills?",
    "What is the importance of cybersecurity?",
    "How can I learn to code effectively?",
    "What are the benefits of meditation?",
    "How do I manage stress at work?",
    "What is sustainable development?",
    "How can I improve my communication skills?",
    "What are the advantages of remote work?"
]

async def create_test_user(db: Session) -> User:
    """Create a test user for feedback simulation"""
    # Check if test user already exists
    user = db.query(User).filter(User.username == "feedback_simulator").first()
    
    if not user:
        user = User(
            username="feedback_simulator",
            email="feedback@simulator.com",
            password_hash="dummy_hash",
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
    return user

async def generate_conversation(db: Session, user: User, prompt: str) -> tuple:
    """Generate a conversation with the model"""
    # Create conversation
    conversation = Conversation(
        user_id=user.id,
        title=prompt[:50] + "..." if len(prompt) > 50 else prompt
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    # Create user message
    user_message = Message(
        conversation_id=conversation.id,
        user_id=user.id,
        content=prompt,
        message_type="user",
        token_count=len(prompt.split())
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)
    
    # Generate AI response
    try:
        response_data = await phi2_manager.generate_response_async(
            prompt=prompt,
            generation_params={
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        )
        
        ai_response_text = response_data["response"]
        generation_time_ms = response_data["generation_time_ms"]
        token_count = response_data["output_tokens"]
        
    except Exception as e:
        print(f"Error generating response: {e}")
        ai_response_text = "I apologize, but I encountered an error generating a response."
        generation_time_ms = 0
        token_count = 0
    
    # Create AI message
    ai_message = Message(
        conversation_id=conversation.id,
        user_id=user.id,
        content=ai_response_text,
        message_type="assistant",
        token_count=token_count
    )
    db.add(ai_message)
    db.commit()
    db.refresh(ai_message)
    
    # Create model response record
    model_response = ModelResponse(
        message_id=ai_message.id,
        model_name="phi-2",
        response_text=ai_response_text,
        generation_time_ms=generation_time_ms,
        token_count=token_count,
        is_primary=True
    )
    db.add(model_response)
    db.commit()
    db.refresh(model_response)
    
    # Create embeddings
    try:
        await create_embedding_for_message(db, user_message)
        await create_embedding_for_response(db, model_response)
    except Exception as e:
        print(f"Error creating embeddings: {e}")
    
    return user_message, ai_message, model_response

def simulate_feedback(db: Session, user: User, model_response: ModelResponse) -> Feedback:
    """Simulate user feedback for a model response"""
    # Simulate different types of feedback based on response quality
    response_text = model_response.response_text.lower()
    
    # Simple heuristics for rating
    rating = 3  # Default neutral rating
    thumbs_up = None
    comment = None
    
    # Positive indicators
    if any(word in response_text for word in ["helpful", "clear", "accurate", "detailed", "comprehensive"]):
        rating = random.choice([4, 5])
        thumbs_up = True
        comment = random.choice([
            "Very helpful response!",
            "Clear and informative.",
            "Exactly what I was looking for.",
            "Great explanation, thank you!"
        ])
    
    # Negative indicators
    elif any(word in response_text for word in ["error", "apologize", "don't know", "unclear"]):
        rating = random.choice([1, 2])
        thumbs_up = False
        comment = random.choice([
            "Not very helpful.",
            "Could be more detailed.",
            "Response seems incomplete.",
            "Needs improvement."
        ])
    
    # Neutral responses
    else:
        rating = random.choice([3, 4])
        thumbs_up = rating >= 3
        
        if random.random() < 0.3:  # 30% chance of comment
            comment = random.choice([
                "Okay response.",
                "Could be better.",
                "Good information.",
                "Thanks for the answer."
            ])
    
    # Create feedback
    feedback = Feedback(
        response_id=model_response.id,
        user_id=user.id,
        rating=rating,
        thumbs_up=thumbs_up,
        comment=comment,
        feedback_type="rating"
    )
    
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    
    return feedback

async def simulate_training_data(
    num_conversations: int = 20,
    feedback_rate: float = 0.8
):
    """Simulate training data by creating conversations and feedback"""
    print(f"Simulating {num_conversations} conversations with {feedback_rate*100}% feedback rate")
    print("=" * 60)
    
    db = SessionLocal()
    
    try:
        # Create test user
        user = await create_test_user(db)
        print(f"Using test user: {user.username}")
        
        # Generate conversations
        total_feedback = 0
        rating_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for i in range(num_conversations):
            # Select random prompt
            prompt = random.choice(TEST_PROMPTS)
            
            print(f"\n{i+1}. Generating conversation for: {prompt[:50]}...")
            
            # Generate conversation
            user_message, ai_message, model_response = await generate_conversation(db, user, prompt)
            
            print(f"   Response length: {len(model_response.response_text)} characters")
            print(f"   Generation time: {model_response.generation_time_ms}ms")
            
            # Simulate feedback
            if random.random() < feedback_rate:
                feedback = simulate_feedback(db, user, model_response)
                total_feedback += 1
                rating_distribution[feedback.rating] += 1
                
                print(f"   Feedback: {feedback.rating}/5 stars, thumbs_up: {feedback.thumbs_up}")
                if feedback.comment:
                    print(f"   Comment: {feedback.comment}")
            else:
                print("   No feedback provided")
            
            # Small delay to simulate realistic timing
            await asyncio.sleep(0.1)
        
        # Summary
        print(f"\n{'='*60}")
        print("Simulation Summary:")
        print(f"Total conversations: {num_conversations}")
        print(f"Total feedback: {total_feedback}")
        print(f"Feedback rate: {total_feedback/num_conversations*100:.1f}%")
        
        print(f"\nRating distribution:")
        for rating, count in rating_distribution.items():
            percentage = count / total_feedback * 100 if total_feedback > 0 else 0
            print(f"  {rating} stars: {count} ({percentage:.1f}%)")
        
        # Calculate average rating
        if total_feedback > 0:
            avg_rating = sum(rating * count for rating, count in rating_distribution.items()) / total_feedback
            print(f"\nAverage rating: {avg_rating:.2f}/5")
        
        print(f"\n✅ Training data simulation completed!")
        print(f"You can now run fine-tuning with: python scripts/run_training.py train")
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        raise
    
    finally:
        db.close()

def show_existing_data():
    """Show existing training data in the database"""
    print("Existing Training Data")
    print("=" * 50)
    
    db = SessionLocal()
    
    try:
        # Count conversations
        total_conversations = db.query(Conversation).count()
        total_messages = db.query(Message).count()
        total_responses = db.query(ModelResponse).count()
        total_feedback = db.query(Feedback).count()
        
        print(f"Total conversations: {total_conversations}")
        print(f"Total messages: {total_messages}")
        print(f"Total model responses: {total_responses}")
        print(f"Total feedback: {total_feedback}")
        
        # Feedback distribution
        if total_feedback > 0:
            print(f"\nFeedback distribution:")
            for rating in range(1, 6):
                count = db.query(Feedback).filter(Feedback.rating == rating).count()
                percentage = count / total_feedback * 100
                print(f"  {rating} stars: {count} ({percentage:.1f}%)")
            
            # Average rating
            avg_rating = db.query(Feedback).with_entities(
                db.func.avg(Feedback.rating)
            ).scalar()
            print(f"\nAverage rating: {avg_rating:.2f}/5")
        
        # Recent feedback
        recent_feedback = db.query(Feedback).order_by(
            Feedback.created_at.desc()
        ).limit(5).all()
        
        if recent_feedback:
            print(f"\nRecent feedback:")
            for feedback in recent_feedback:
                print(f"  Rating: {feedback.rating}/5, Thumbs up: {feedback.thumbs_up}")
                if feedback.comment:
                    print(f"    Comment: {feedback.comment[:50]}...")
        
    except Exception as e:
        print(f"Error showing data: {e}")
    
    finally:
        db.close()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulate feedback for training")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Simulate command
    simulate_parser = subparsers.add_parser("simulate", help="Simulate training data")
    simulate_parser.add_argument("--conversations", type=int, default=20, help="Number of conversations")
    simulate_parser.add_argument("--feedback-rate", type=float, default=0.8, help="Feedback rate (0-1)")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show existing data")
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        asyncio.run(simulate_training_data(
            num_conversations=args.conversations,
            feedback_rate=args.feedback_rate
        ))
    elif args.command == "show":
        show_existing_data()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()