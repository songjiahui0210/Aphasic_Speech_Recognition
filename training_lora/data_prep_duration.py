import argparse
from transformers import WhisperFeatureExtractor
import pandas as pd
from datasets import Dataset
import os
import soundfile as sf
import numpy as np

def process_audio_file(batch):
    audio_data = []
    missing_files = []
    for folder_name, file_cut in zip(batch['folder_name'], batch['file_cut']):
        audio_file_path = os.path.join("../../data_processed/audios", folder_name, file_cut)
        if os.path.exists(audio_file_path):
            try:
                audio, sample_rate = sf.read(audio_file_path)
                audio_data.append({'array': audio.tolist() if isinstance(audio, np.ndarray) else audio, 'sampling_rate': sample_rate})
            except Exception as e:
                print(f"Error loading {audio_file_path}: {e}")
                audio_data.append(None)
        else:
            print(f"File not found: {audio_file_path}")
            audio_data.append(None)
            missing_files.append(audio_file_path)
    return {'audio': audio_data, 'missing_files': missing_files}

def prepare_dataset(dataset, feature_extractor):
    def prepare(batch):
        processed_data = {'input_features': []}
        for audio_info in batch['audio']:
            if audio_info and audio_info['array'] is not None:
                try:
                    input_features = feature_extractor(audio_info['array'], sampling_rate=audio_info['sampling_rate']).input_features[0]
                    processed_data['input_features'].append(input_features)
                except Exception as e:
                    print(f"Error processing audio: {e}")
                    processed_data['input_features'].append(None)
            else:
                processed_data['input_features'].append(None)
        return processed_data

    return dataset.map(prepare, batched=True, batch_size=50)

def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset with Whisper model.")
    parser.add_argument("model_size", type=str, choices=["small", "large"],
                        help="Size of the Whisper model to use.")
    args = parser.parse_args()

    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{args.model_size}")
    csv_file_path = '../../data_processed/dataset_splitted_by_duration.csv'
    df = pd.read_csv(csv_file_path)

    # Split data based on 'split' column
    train_data = df[df['split'] == 'train']
    eval_data = df[df['split'] == 'eval']
    test_data = df[df['split'] == 'test']

    # Convert dataframes to datasets
    train_dataset = Dataset.from_pandas(train_data)
    eval_dataset = Dataset.from_pandas(eval_data)
    test_dataset = Dataset.from_pandas(test_data)

    # Process each dataset
    train_processed = process_audio_file(train_dataset)
    eval_processed = process_audio_file(eval_dataset)
    test_processed = process_audio_file(test_dataset)

    # Prepare datasets for feature extraction
    train_features = prepare_dataset(train_processed, feature_extractor)
    eval_features = prepare_dataset(eval_processed, feature_extractor)
    test_features = prepare_dataset(test_processed, feature_extractor)

    # Save processed datasets
    train_features.save_to_disk(f"../../data_processed/train_dataset_{args.model_size}")
    eval_features.save_to_disk(f"../../data_processed/eval_dataset_{args.model_size}")
    test_features.save_to_disk(f"../../data_processed/test_dataset_{args.model_size}")

    print("Dataset processing completed and saved.")

if __name__ == "__main__":
    main()