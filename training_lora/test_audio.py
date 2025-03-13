import argparse
import pandas as pd
import os
import soundfile as sf
import numpy as np

from datasets import Dataset, load_from_disk
from transformers import WhisperProcessor

def process_audio_file(batch):
    """Load audio files and ensure they exist."""
    
    if batch["folder_name"] != "ACWT":
        return {"audio": None}  # Skip processing non-ACWT files

    audio_file_path = os.path.join("../../data_processed/audios/ACWT", batch["file_cut"])
    
    print(f"Processing audio: {audio_file_path}")  # Debug Checkpoint

    if os.path.exists(audio_file_path):
        try:
            audio, sample_rate = sf.read(audio_file_path)
            audio_list = audio.tolist() if isinstance(audio, np.ndarray) else audio
            return {
                "audio": {
                    "array": audio_list,
                    "sampling_rate": int(sample_rate)
                }
            }
        except Exception as e:
            print(f"Error reading {audio_file_path}: {e}")
            return {"audio": None}
    else:
        print(f"Missing file: {audio_file_path}")  
        return {"audio": None}

def prepare_dataset(dataset, processor):
    """Prepare dataset by extracting audio features and tokenizing transcriptions."""
    
    def prepare(batch):
        processed_data = {"input_features": [], "labels": []}

        for i, audio_info in enumerate(batch.get("audio", [])):
            if audio_info and "array" in audio_info and audio_info["array"] is not None:
                print(f" Extracting features for sample {i}")  # Checkpoint 3
                try:
                    feats = processor.feature_extractor(
                        audio_info["array"],
                        sampling_rate=audio_info["sampling_rate"]
                    ).input_features[0]
                    processed_data["input_features"].append(feats)
                except Exception as e:
                    print(f" Feature extraction error for sample {i}: {e}")
                    processed_data["input_features"].append(None)
            else:
                print(f" Missing or corrupted audio for sample {i}")  #  Checkpoint 4
                processed_data["input_features"].append(None)

        # Tokenizing text
        text_list = batch.get("transcriptions", [])
        if isinstance(text_list, list):
            for text in text_list:
                if text is not None and isinstance(text, str):
                    tokenized = processor.tokenizer(text)
                    processed_data["labels"].append(tokenized["input_ids"])
                else:
                    processed_data["labels"].append([])
        else:
            processed_data["labels"] = [[] for _ in range(len(batch.get("audio", [])))]

        return processed_data

    return dataset.map(
        prepare,
        batched=True,
        batch_size=5,
        num_proc=4
    )

def save_missing_files(missing_files):
    """Save a list of missing audio files."""
    if missing_files:
        missing_files_df = pd.DataFrame(missing_files, columns=["missing_file_path"])
        missing_files_df.to_csv("../../data_processed/missing_audio_files.csv", index=False)
        print(" Missing audio files saved.")
    else:
        print(" No missing audio files detected.")

def main():
    """Main function to load, process, and save dataset."""
    
    parser = argparse.ArgumentParser(description="Prepare a dataset with Whisper model.")
    parser.add_argument("model_size", type=str, choices=["small", "large"], help="Size of the Whisper model to use.")
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{args.model_size}")

    csv_file_path = '../../data_processed/set1_w_cohort.csv'
    df = pd.read_csv(csv_file_path)
    columns_to_drop = ['name','sex','age','file','WAB_AQ','aphasia_type','WAB_AQ_category','fluency_speech','original_file_length','difference','name_extracted_from_filename']
    df = df.drop(columns=columns_to_drop)
    print(f" CSV loaded with {len(df)} rows.")  #  Checkpoint 5

    # Filter valid speakers
    df['utterance_duration'] = (df['mark_end'] - df['mark_start']).astype(int)
    speaker_durations = df.groupby('name_unique_speaker')['utterance_duration'].sum().reset_index()
    valid_speakers = speaker_durations[speaker_durations['utterance_duration'] > 480000]['name_unique_speaker']
    df_filtered = df[df['name_unique_speaker'].isin(valid_speakers)]
    print(f" Filtered dataset has {len(df_filtered)} rows.")  #  Checkpoint 6

    # Convert to Hugging Face dataset
    dataset = Dataset.from_pandas(df_filtered)
    dataset = dataset.map(process_audio_file)
    dataset = prepare_dataset(dataset, processor)

    # Save processed dataset
    dataset.save_to_disk(f"/scratch/song.jiahui/data_processed/processed_dataset_acwt")
    print(" Dataset processing complete!")  #  Checkpoint 7

if __name__ == "__main__":
    main()