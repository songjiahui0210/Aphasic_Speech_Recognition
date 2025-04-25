#!/usr/bin/env python3
"""
Prepare data for the Aphasic Speech Recognition demo.
This script handles the preprocessing of audio files and transcriptions
for a small subset of data to be used in the demo.

The script:
1. Loads a subset of data from a CSV file
2. Processes the audio files to extract features
3. Tokenizes the transcriptions
4. Saves the processed data as datasets ready for training

Usage:
    python prepare_data_for_demo.py --csv_path demo_data/demo_subset.csv --model_size small
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset
import soundfile as sf
from transformers import WhisperProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for the Aphasic Speech Recognition demo")
    parser.add_argument("--csv_path", type=str, default="demo_data/demo_subset.csv",
                      help="Path to the subset CSV file")
    parser.add_argument("--audio_root", type=str, default="demo_data/audios",
                      help="Root directory containing audio files")
    parser.add_argument("--model_size", type=str, default="small",
                      choices=["tiny", "base", "small", "medium", "large"],
                      help="Size of the Whisper model")
    parser.add_argument("--output_dir", type=str, default="demo_data",
                      help="Directory to save processed datasets")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                      help="Ratio of data to use for training (vs. validation)")
    
    return parser.parse_args()

def process_audio_files(df, audio_root, processor):
    """Process audio files and extract features."""
    print("Processing audio files...")
    
    processed_data = []
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = os.path.join(audio_root, row['folder_name'], row['file_cut'])
        
        # Skip if audio file doesn't exist
        if not os.path.exists(audio_path):
            skipped += 1
            continue
        
        try:
            # Load audio
            audio, sample_rate = sf.read(audio_path)
            
            # Extract features
            input_features = processor.feature_extractor(
                audio, 
                sampling_rate=sample_rate
            ).input_features[0]
            
            # Tokenize text
            if isinstance(row['transcriptions'], str) and len(row['transcriptions'].strip()) > 0:
                tokenized = processor.tokenizer(row['transcriptions'])
                labels = tokenized.input_ids
            else:
                labels = []
            
            # Store processed sample
            processed_data.append({
                'file_path': audio_path,
                'input_features': input_features,
                'labels': labels,
                'transcription': row['transcriptions'] if isinstance(row['transcriptions'], str) else "",
                'file_cut': row['file_cut'],
                'folder_name': row['folder_name']
            })
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            skipped += 1
    
    print(f"Processed {len(processed_data)} files, skipped {skipped} files")
    return processed_data

def main():
    args = parse_args()
    
    print(f"=== Preparing Data for Whisper-{args.model_size} Demo ===")
    
    # Load the processor
    print(f"Loading WhisperProcessor for model size: {args.model_size}")
    processor = WhisperProcessor.from_pretrained(
        f"openai/whisper-{args.model_size}", 
        language="en", 
        task="transcribe"
    )
    
    # Load the CSV
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file {args.csv_path} not found")
        return
    
    df = pd.read_csv(args.csv_path)
    print(f"Loaded CSV with {len(df)} samples")
    
    # Process the audio files
    processed_data = process_audio_files(df, args.audio_root, processor)
    
    if not processed_data:
        print("No data was processed. Exiting.")
        return
    
    # Convert to dataset
    dataset = Dataset.from_dict({
        'file_path': [item['file_path'] for item in processed_data],
        'input_features': [item['input_features'] for item in processed_data],
        'labels': [item['labels'] for item in processed_data],
        'transcription': [item['transcription'] for item in processed_data],
        'file_cut': [item['file_cut'] for item in processed_data],
        'folder_name': [item['folder_name'] for item in processed_data]
    })
    
    # Split into train and validation
    train_test_split = dataset.train_test_split(
        test_size=(1.0 - args.train_ratio),
        seed=42
    )
    
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    print(f"Split data into {len(train_dataset)} train and {len(eval_dataset)} validation samples")
    
    # Save the datasets
    train_output = os.path.join(args.output_dir, f"train_dataset_{args.model_size}")
    eval_output = os.path.join(args.output_dir, f"eval_dataset_{args.model_size}")
    
    train_dataset.save_to_disk(train_output)
    eval_dataset.save_to_disk(eval_output)
    
    print(f"Saved processed datasets to:")
    print(f"  - {train_output}")
    print(f"  - {eval_output}")
    
    print("=== Data Preparation Complete ===")

if __name__ == "__main__":
    main()