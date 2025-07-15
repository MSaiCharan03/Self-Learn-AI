import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from datasets import Dataset
import os
import json
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from sqlalchemy.orm import Session
from ..models.conversation import Message, ModelResponse
from ..models.feedback import Feedback, TrainingSession, ModelComparison
from ..utils.config import settings
from .llm_manager import phi2_manager

logger = logging.getLogger(__name__)

class FineTuningDataProcessor:
    """Process training data from database for fine-tuning"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 1024
        
    def extract_conversation_pairs(self, db: Session, min_rating: float = 3.0) -> List[Dict]:
        """Extract conversation pairs from database with quality filtering"""
        # Get messages with high-rated responses
        query = db.query(Message, ModelResponse, Feedback).join(
            ModelResponse, Message.id == ModelResponse.message_id
        ).join(
            Feedback, ModelResponse.id == Feedback.response_id
        ).filter(
            Feedback.rating >= min_rating,
            ModelResponse.model_name == "phi-2"
        )
        
        conversation_pairs = []
        
        for message, response, feedback in query:
            # Get conversation context
            context_messages = db.query(Message).filter(
                Message.conversation_id == message.conversation_id,
                Message.created_at < message.created_at
            ).order_by(Message.created_at.desc()).limit(5).all()
            
            # Build context
            context = ""
            for ctx_msg in reversed(context_messages):
                role = "Human" if ctx_msg.message_type == "user" else "Assistant"
                context += f"{role}: {ctx_msg.content}\n"
            
            # Create training example
            training_example = {
                "context": context,
                "input": message.content,
                "output": response.response_text,
                "rating": feedback.rating,
                "conversation_id": str(message.conversation_id),
                "created_at": message.created_at.isoformat()
            }
            
            conversation_pairs.append(training_example)
        
        return conversation_pairs
    
    def extract_preference_pairs(self, db: Session) -> List[Dict]:
        """Extract preference pairs from model comparisons"""
        comparisons = db.query(ModelComparison).filter(
            ModelComparison.winner.isnot(None)
        ).all()
        
        preference_pairs = []
        
        for comparison in comparisons:
            # Get the responses
            phi2_response = db.query(ModelResponse).filter(
                ModelResponse.id == comparison.phi2_response_id
            ).first()
            
            external_response = db.query(ModelResponse).filter(
                ModelResponse.id == comparison.external_response_id
            ).first()
            
            if phi2_response and external_response:
                # Create preference example
                preference_example = {
                    "prompt": comparison.prompt_text,
                    "chosen": phi2_response.response_text if comparison.winner == "phi2" else external_response.response_text,
                    "rejected": external_response.response_text if comparison.winner == "phi2" else phi2_response.response_text,
                    "winner": comparison.winner,
                    "created_at": comparison.created_at.isoformat()
                }
                
                preference_pairs.append(preference_example)
        
        return preference_pairs
    
    def extract_negative_examples(self, db: Session) -> List[Dict]:
        """Extract negative examples from low-rated responses"""
        # Get messages with low-rated responses
        query = db.query(Message, ModelResponse, Feedback).join(
            ModelResponse, Message.id == ModelResponse.message_id
        ).join(
            Feedback, ModelResponse.id == Feedback.response_id
        ).filter(
            Feedback.rating <= 2.0,
            ModelResponse.model_name == "phi-2"
        )
        
        negative_examples = []
        
        for message, response, feedback in query:
            negative_example = {
                "input": message.content,
                "bad_output": response.response_text,
                "rating": feedback.rating,
                "feedback_comment": feedback.comment,
                "created_at": message.created_at.isoformat()
            }
            
            negative_examples.append(negative_example)
        
        return negative_examples
    
    def create_training_dataset(self, conversation_pairs: List[Dict]) -> Dataset:
        """Create HuggingFace dataset from conversation pairs"""
        formatted_examples = []
        
        for pair in conversation_pairs:
            # Format as a conversation
            prompt = f"System: You are a helpful, harmless, and honest AI assistant.\n\n"
            if pair["context"]:
                prompt += f"{pair['context']}"
            prompt += f"Human: {pair['input']}\nAssistant:"
            
            # Full text for training
            full_text = f"{prompt} {pair['output']}"
            
            # Tokenize
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            # Create labels (same as input_ids for causal LM)
            labels = tokenized["input_ids"].copy()
            
            # Mask prompt tokens so model only learns to predict response
            prompt_tokens = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            prompt_length = len(prompt_tokens["input_ids"])
            labels[:prompt_length] = [-100] * prompt_length  # Ignore prompt in loss
            
            formatted_examples.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels,
                "rating": pair["rating"]
            })
        
        return Dataset.from_list(formatted_examples)

class LoRATrainer:
    """Low-Rank Adaptation (LoRA) fine-tuning for Phi-2"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.phi2_model_path
        self.device = "cpu"  # Force CPU training
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def setup_model_and_tokenizer(self):
        """Setup base model and tokenizer for LoRA training"""
        logger.info("Setting up model and tokenizer for LoRA training...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            cache_dir=self.model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            cache_dir=self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("Model and tokenizer setup complete")
    
    def create_lora_config(self, rank: int = 8, alpha: int = 32) -> LoraConfig:
        """Create LoRA configuration"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Phi-2 attention layers
            bias="none",
            inference_mode=False
        )
    
    def setup_lora_model(self, lora_config: LoraConfig):
        """Setup LoRA model"""
        logger.info("Setting up LoRA model...")
        
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info("LoRA model setup complete")
    
    def train(
        self,
        train_dataset: Dataset,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 1,  # Small batch size for CPU
        learning_rate: float = 5e-5,
        save_steps: int = 100,
        eval_steps: int = 100,
        warmup_steps: int = 10,
        logging_steps: int = 10
    ) -> Dict[str, Any]:
        """Train the LoRA model"""
        logger.info(f"Starting LoRA training with {len(train_dataset)} examples...")
        
        # Training arguments optimized for CPU
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,  # Simulate larger batch size
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Disable multiprocessing for CPU
            fp16=False,  # Disable mixed precision for CPU
            optim="adamw_torch",
            weight_decay=0.01,
            max_grad_norm=1.0,
            lr_scheduler_type="linear",
            seed=42
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Split dataset for eval
        split_dataset = train_dataset.train_test_split(test_size=0.1)
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Start training
        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        # Save the model
        trainer.save_model()
        trainer.save_state()
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
            "training_time": training_time,
            "output_dir": output_dir,
            "num_epochs": num_epochs,
            "num_examples": len(train_dataset)
        }

class FeedbackBasedTrainer:
    """Training pipeline based on user feedback"""
    
    def __init__(self):
        self.data_processor = None
        self.lora_trainer = None
        
    def prepare_training_data(self, db: Session, min_examples: int = 10) -> Tuple[Dataset, Dict]:
        """Prepare training data from database"""
        logger.info("Preparing training data from database...")
        
        # Initialize data processor
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            cache_dir=settings.phi2_model_path,
            trust_remote_code=True
        )
        
        self.data_processor = FineTuningDataProcessor(tokenizer)
        
        # Extract conversation pairs
        conversation_pairs = self.data_processor.extract_conversation_pairs(db)
        logger.info(f"Found {len(conversation_pairs)} conversation pairs")
        
        # Extract preference pairs
        preference_pairs = self.data_processor.extract_preference_pairs(db)
        logger.info(f"Found {len(preference_pairs)} preference pairs")
        
        # Extract negative examples
        negative_examples = self.data_processor.extract_negative_examples(db)
        logger.info(f"Found {len(negative_examples)} negative examples")
        
        if len(conversation_pairs) < min_examples:
            raise ValueError(f"Not enough training examples. Need at least {min_examples}, got {len(conversation_pairs)}")
        
        # Create training dataset
        train_dataset = self.data_processor.create_training_dataset(conversation_pairs)
        
        # Statistics
        stats = {
            "total_conversation_pairs": len(conversation_pairs),
            "total_preference_pairs": len(preference_pairs),
            "total_negative_examples": len(negative_examples),
            "average_rating": sum(pair["rating"] for pair in conversation_pairs) / len(conversation_pairs),
            "dataset_size": len(train_dataset)
        }
        
        logger.info(f"Training dataset prepared with {len(train_dataset)} examples")
        
        return train_dataset, stats
    
    def start_training_session(
        self,
        db: Session,
        training_type: str = "lora",
        parameters: Dict = None
    ) -> TrainingSession:
        """Start a new training session"""
        logger.info(f"Starting {training_type} training session...")
        
        # Create training session record
        session = TrainingSession(
            model_name="phi-2",
            training_type=training_type,
            status="running",
            parameters=parameters or {},
            start_time=datetime.utcnow()
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return session
    
    def run_lora_training(
        self,
        db: Session,
        session: TrainingSession,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        rank: int = 8,
        alpha: int = 32
    ) -> Dict[str, Any]:
        """Run LoRA fine-tuning"""
        try:
            # Prepare training data
            train_dataset, data_stats = self.prepare_training_data(db)
            
            # Update session with data size
            session.data_size = data_stats["dataset_size"]
            session.parameters.update(data_stats)
            db.commit()
            
            # Setup LoRA trainer
            self.lora_trainer = LoRATrainer()
            self.lora_trainer.setup_model_and_tokenizer()
            
            # Create LoRA config
            lora_config = self.lora_trainer.create_lora_config(rank=rank, alpha=alpha)
            self.lora_trainer.setup_lora_model(lora_config)
            
            # Output directory
            output_dir = os.path.join(settings.phi2_model_path, f"lora_checkpoint_{session.id}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Train model
            training_results = self.lora_trainer.train(
                train_dataset=train_dataset,
                output_dir=output_dir,
                num_epochs=num_epochs,
                learning_rate=learning_rate
            )
            
            # Update session
            session.status = "completed"
            session.end_time = datetime.utcnow()
            session.model_checkpoint_path = output_dir
            session.loss_metrics = {
                "final_train_loss": training_results["train_loss"],
                "train_runtime": training_results["train_runtime"],
                "samples_per_second": training_results["train_samples_per_second"]
            }
            
            db.commit()
            
            logger.info(f"LoRA training completed successfully. Checkpoint saved to {output_dir}")
            
            return {
                "success": True,
                "session_id": str(session.id),
                "output_dir": output_dir,
                "training_results": training_results,
                "data_stats": data_stats
            }
            
        except Exception as e:
            logger.error(f"LoRA training failed: {e}")
            
            # Update session with error
            session.status = "failed"
            session.end_time = datetime.utcnow()
            session.error_log = str(e)
            db.commit()
            
            return {
                "success": False,
                "error": str(e),
                "session_id": str(session.id)
            }
    
    def evaluate_model_improvement(
        self,
        db: Session,
        checkpoint_path: str,
        test_prompts: List[str] = None
    ) -> Dict[str, Any]:
        """Evaluate model improvement after training"""
        logger.info("Evaluating model improvement...")
        
        if test_prompts is None:
            test_prompts = [
                "What is artificial intelligence?",
                "How can I improve my productivity?",
                "Explain machine learning in simple terms.",
                "What are the benefits of renewable energy?",
                "How do I learn a new programming language?"
            ]
        
        results = {
            "original_model": [],
            "fine_tuned_model": [],
            "improvement_scores": []
        }
        
        try:
            # Load original model responses
            for prompt in test_prompts:
                response = phi2_manager.generate_response(prompt)
                results["original_model"].append({
                    "prompt": prompt,
                    "response": response["response"],
                    "generation_time": response["generation_time_ms"]
                })
            
            # TODO: Load fine-tuned model and generate responses
            # This would require loading the LoRA checkpoint
            # For now, we'll simulate improvement
            
            logger.info("Model evaluation completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {"error": str(e)}

# Global fine-tuning manager
fine_tuning_manager = FeedbackBasedTrainer()