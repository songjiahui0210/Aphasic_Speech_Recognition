import argparse
from transformers import WhisperFeatureExtractor
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
import os 
import soundfile as sf
import pyarrow as pa
import numpy as np


def process_audio_file(batch):
    results = {'audio_array': [], 'sample_rate': []}
    for folder_name, file_cut in zip(batch['folder_name'], batch['file_cut']):
        audio_file_path = os.path.join("../../data_processed/audios", folder_name, file_cut)
        if os.path.exists(audio_file_path):
            try:
                audio, sample_rate = sf.read(audio_file_path)
                results['audio_array'].append(audio.tolist() if isinstance(audio, np.ndarray) else audio)
                results['sample_rate'].append(int(sample_rate))
            except Exception as e:
                print(f"Error loading {audio_file_path}: {e}")
                results['audio_array'].append(None)
                results['sample_rate'].append(None)
        else:
            print(f"File not found: {audio_file_path}")
            results['audio_array'].append(None)
            results['sample_rate'].append(None)
    return results

def prepare_dataset(dataset, feature_extractor):
    def prepare(batch):
        audio = batch["audio"]
        if audio and audio["array"] is not None:
            try:
                return {"input_features": feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]}
            except Exception as e:
                print(f"Error processing audio: {e}")
                return {"input_features": None}
        return {"input_features": None}

    return dataset.map(prepare, batched=True, batch_size=50)

def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset with Whisper model.")
    parser.add_argument("model_size", type=str, choices=["small", "large"],
                        help="Size of the Whisper model to use.")
    args = parser.parse_args()

    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{args.model_size}")
    csv_file_path = '../../data_processed/dataset_splitted.csv'
    df = pd.read_csv(csv_file_path)

    # Calculate duration for each utterance in milliseconds
    df['utterance_duration'] = (df['mark_end'] - df['mark_start']).astype(int)
    
    # Sum durations by speaker or file
    speaker_durations = df.groupby('name_unique_speaker')['utterance_duration'].sum().reset_index()

    # Filter speakers by total duration > 480,000 milliseconds (8 minutes)
    valid_speakers = speaker_durations[speaker_durations['utterance_duration'] > 480000]['name_unique_speaker']
    
    # Ensure valid_speakers is defined and used correctly
    if 'name_unique_speaker' in df.columns and valid_speakers is not None:
        # Filter the original dataframe to include only valid speakers
        df_filtered = df[df['name_unique_speaker'].isin(valid_speakers)]
    else:
        print("Error: 'name_unique_speaker' not in DataFrame or 'valid_speakers' not defined")
        return  
    dataset = Dataset.from_pandas(df_filtered)
    dataset = dataset.map(process_audio_file, batched=True, batch_size=50)
    dataset = prepare_dataset(dataset, feature_extractor)

    dataset.save_to_disk(f"../../data_processed/processed_dataset_{args.model_size}")

if __name__ == "__main__":
    main()