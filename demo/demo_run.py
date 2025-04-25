#!/usr/bin/env python3
"""
Demo script for Aphasic Speech Recognition with Whisper and LoRA.
This script demonstrates the key components of training a Whisper model 
with Low-Rank Adaptation (LoRA) for aphasic speech recognition.

The script will:
1. Prepare a small subset of data for demo purposes
2. Configure and initialize a Whisper model with LoRA adapters
3. Start training for a few steps to demonstrate the process
4. Show how to evaluate a model with a test audio file

Usage:
    python demo_run.py --model_size small --demo_steps 10 --data_subset 100
"""

import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from peft import LoraConfig, get_peft_model
import soundfile as sf
import numpy as np

# Add the training_lora directory to path to import modules
import sys
sys.path.append("../training_lora")
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from compute_metrics import compute_metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demo for Aphasic Speech Recognition with Whisper LoRA")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium"],
                      help="Size of the Whisper model (default: small)")
    parser.add_argument("--data_subset", type=int, default=100,
                      help="Number of samples to use for the demo (default: 100)")
    parser.add_argument("--demo_steps", type=int, default=10,
                      help="Number of training steps to run (default: 10)")
    parser.add_argument("--lora_r", type=int, default=8,
                      help="Rank of LoRA adapters (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                      help="Alpha scaling factor for LoRA (default: 16)")
    parser.add_argument("--test_audio", type=str, 
                      default="../../data_processed/audios/ACWT/ACWT01a_144.813_2.78.wav",
                      help="Path to a test audio file")
    
    return parser.parse_args()

def prepare_demo_data(args, base_csv_path="../../data_processed/set1_w_cohort.csv"):
    """
    Prepare a small subset of data for the demo.
    Creates train and validation datasets with processed features.
    """
    print("=== Preparing Demo Dataset ===")
    
    # Create demo data directories if they don't exist
    os.makedirs("demo_data", exist_ok=True)
    
    # Load the CSV data
    if os.path.exists(base_csv_path):
        df = pd.read_csv(base_csv_path)
        print(f"Loaded data from {base_csv_path}, total rows: {len(df)}")
    else:
        print(f"Warning: {base_csv_path} not found. Creating dummy dataset.")
        # Create a dummy dataframe if the actual data doesn't exist
        df = pd.DataFrame({
            "folder_name": ["ACWT"] * args.data_subset,
            "file_cut": [f"dummy_audio_{i}.wav" for i in range(args.data_subset)],
            "transcriptions": ["this is a dummy transcription"] * args.data_subset,
            "mark_start": [0] * args.data_subset,
            "mark_end": [1000] * args.data_subset
        })
    
    # Take a subset for the demo
    df_subset = df.head(args.data_subset)
    
    # Split into train (80%) and validation (20%)
    train_size = int(0.8 * len(df_subset))
    train_df = df_subset[:train_size]
    val_df = df_subset[train_size:]
    
    print(f"Created demo subset: {len(train_df)} train samples, {len(val_df)} validation samples")
    
    # Convert to HF datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Process the data (in practice, this would preprocess audio files)
    # For the demo, we'll create dummy input features
    def add_dummy_features(example):
        # Create a dummy spectrogram of shape [80, 3000]
        example["input_features"] = np.random.randn(80, 3000).astype(np.float32)
        # Create dummy token ids
        example["labels"] = [101, 102, 103, 104, 105, 106, 107, 108]
        return example
    
    print("Processing train dataset...")
    train_dataset = train_dataset.map(add_dummy_features)
    print("Processing validation dataset...")
    val_dataset = val_dataset.map(add_dummy_features)
    
    # Save the datasets
    train_path = "demo_data/train_dataset"
    val_path = "demo_data/val_dataset"
    
    train_dataset.save_to_disk(train_path)
    val_dataset.save_to_disk(val_path)
    
    print(f"Saved demo datasets to {train_path} and {val_path}")
    
    return train_path, val_path

def load_real_audio(audio_path):
    """Load and process a real audio file for testing."""
    if not os.path.exists(audio_path):
        print(f"Warning: Test audio file {audio_path} not found.")
        return None, None
    
    try:
        # Load audio file
        audio, sample_rate = sf.read(audio_path)
        print(f"Loaded audio file: {audio_path}")
        print(f"Audio shape: {audio.shape}, Sample rate: {sample_rate}Hz, Duration: {len(audio)/sample_rate:.2f}s")
        return audio, sample_rate
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None

def initialize_model(args):
    """Initialize the Whisper model with LoRA adapters."""
    print(f"=== Initializing Whisper-{args.model_size} with LoRA ===")
    
    # Load base model
    model_id = f"openai/whisper-{args.model_size}"
    
    print(f"Loading model: {model_id}")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    processor = WhisperProcessor.from_pretrained(model_id, language="en", task="transcribe")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,                   
        lora_alpha=args.lora_alpha,         
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )
    
    # Apply LoRA adapter
    whisper_model = get_peft_model(whisper_model, lora_config)
    
    # Print trainable parameters
    whisper_model.print_trainable_parameters()
    
    return whisper_model, processor

def setup_trainer(model, processor, train_dataset, val_dataset, args):
    """Set up the trainer with datasets and training arguments."""
    print("=== Setting Up Trainer ===")
    
    # Create output directory for the demo
    output_dir = "demo_data/demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        num_train_epochs=1,
        max_steps=args.demo_steps,  # Only run for a few steps for the demo
        
        # Evaluation and checkpointing
        evaluation_strategy="steps",
        eval_steps=5,
        save_steps=5,
        logging_steps=1,
        
        # Other settings
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",  # Disable reporting for the demo
        predict_with_generate=True,
        generation_max_length=225
    )
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=lambda p: compute_metrics(p, processor.tokenizer)
    )
    
    return trainer

def test_transcription(model, processor, audio_path):
    """Test transcription on a single audio file."""
    print(f"=== Testing Transcription on {audio_path} ===")
    
    audio, sample_rate = load_real_audio(audio_path)
    if audio is None:
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Process audio
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    
    # Generate transcription
    print("Generating transcription...")
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    
    # Decode the predicted ids
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    print(f"Transcription: {transcription}")

def main():
    """Main function for the demo."""
    args = parse_args()
    
    print("\n===== Aphasic Speech Recognition Demo =====")
    print(f"Model size: whisper-{args.model_size}")
    print(f"LoRA parameters: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Using a subset of {args.data_subset} samples")
    print(f"Will run for {args.demo_steps} training steps\n")
    
    # Prepare demo data
    train_path, val_path = prepare_demo_data(args)
    
    # Load datasets
    train_dataset = load_from_disk(train_path)
    val_dataset = load_from_disk(val_path)
    
    # Initialize model
    model, processor = initialize_model(args)
    
    # Set up trainer
    trainer = setup_trainer(model, processor, train_dataset, val_dataset, args)
    
    # Start training
    print("\n=== Starting Demo Training ===")
    print(f"Running for {args.demo_steps} steps, this may take a few minutes...")
    trainer.train()
    
    print("\n=== Demo Training Complete ===")
    print("A full training run would continue for many more steps.")
    
    # Test transcription on a real audio file
    test_transcription(model, processor, args.test_audio)
    
    print("\n=== Demo Complete ===")
    print("To perform a full training run, use the training_lora/train_lora.py script.")
    print("For personalized adaptation, use the two-stage approach described in the README.")

if __name__ == "__main__":
    main()