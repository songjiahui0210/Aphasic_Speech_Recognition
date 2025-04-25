#!/usr/bin/env python3
"""
Create a smaller subset of the aphasic speech dataset for demonstration purposes.
This script selects a subset of speakers and their audio files to create a manageable
demo dataset without having to process the entire corpus.

Usage:
    python create_data_subset.py --num_speakers 5 --samples_per_speaker 20 --output_dir demo_data
"""

import os
import argparse
import pandas as pd
import shutil
from tqdm import tqdm
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Create a subset of data for demo purposes")
    parser.add_argument("--csv_path", type=str, default="../../data_processed/set1_w_cohort.csv",
                      help="Path to the full dataset CSV")
    parser.add_argument("--num_speakers", type=int, default=5,
                      help="Number of speakers to include in the subset")
    parser.add_argument("--samples_per_speaker", type=int, default=20,
                      help="Maximum number of samples per speaker")
    parser.add_argument("--output_dir", type=str, default="demo_data",
                      help="Directory to save the subset")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    return parser.parse_args()

def create_subset(args):
    print(f"Creating data subset with {args.num_speakers} speakers and up to {args.samples_per_speaker} samples per speaker")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "audios"), exist_ok=True)
    
    # Load the full dataset
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file {args.csv_path} not found.")
        return
    
    df = pd.read_csv(args.csv_path)
    print(f"Loaded full dataset with {len(df)} samples")
    
    # Get unique speakers
    speaker_column = 'name_unique_speaker' if 'name_unique_speaker' in df.columns else 'name'
    if speaker_column not in df.columns:
        print(f"Error: Speaker column '{speaker_column}' not found in CSV.")
        return
    
    unique_speakers = df[speaker_column].unique()
    print(f"Found {len(unique_speakers)} unique speakers")
    
    # Randomly select a subset of speakers
    selected_speakers = random.sample(list(unique_speakers), min(args.num_speakers, len(unique_speakers)))
    print(f"Selected {len(selected_speakers)} speakers: {selected_speakers}")
    
    # Filter the dataframe to only include selected speakers
    df_subset = df[df[speaker_column].isin(selected_speakers)]
    
    # For each speaker, select a limited number of samples
    final_subset = []
    for speaker in selected_speakers:
        speaker_samples = df_subset[df_subset[speaker_column] == speaker]
        selected_samples = speaker_samples.sample(min(args.samples_per_speaker, len(speaker_samples)))
        final_subset.append(selected_samples)
    
    # Combine all selected samples into a single dataframe
    subset_df = pd.concat(final_subset)
    print(f"Created subset with {len(subset_df)} total samples")
    
    # Save the subset CSV
    output_csv = os.path.join(args.output_dir, "demo_subset.csv")
    subset_df.to_csv(output_csv, index=False)
    print(f"Saved subset metadata to {output_csv}")
    
    # Copy the audio files for the subset
    audio_root = "../../data_processed/audios"
    print("Copying audio files...")
    
    for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
        src_path = os.path.join(audio_root, row['folder_name'], row['file_cut'])
        
        # Create destination folder if it doesn't exist
        dst_folder = os.path.join(args.output_dir, "audios", row['folder_name'])
        os.makedirs(dst_folder, exist_ok=True)
        
        dst_path = os.path.join(dst_folder, row['file_cut'])
        
        # Copy the file if it exists
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: Source file not found: {src_path}")
    
    print(f"Data subset creation complete. Files saved to {args.output_dir}")
    return output_csv

def main():
    args = parse_args()
    create_subset(args)

if __name__ == "__main__":
    main()