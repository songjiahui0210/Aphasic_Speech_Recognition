import argparse
from transformers import WhisperFeatureExtractor
import pandas as pd
from datasets import Dataset, load_from_disk
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
                missing_files.append(None)  # Append None or a placeholder for non-missing files
            except Exception as e:
                print(f"Error loading {audio_file_path}: {e}")
                audio_data.append(None)
                missing_files.append(audio_file_path)  
        else:
            print(f"File not found: {audio_file_path}")
            audio_data.append(None)
            missing_files.append(audio_file_path) 

    return {'audio': audio_data, 'missing_files': missing_files}

def prepare_dataset(dataset, feature_extractor):
    def prepare(batch):
        processed_data = {'input_features': []}
        if 'audio' in batch and isinstance(batch['audio'], list):
            for audio_info in batch['audio']:
                if audio_info and 'array' in audio_info and audio_info['array'] is not None:
                    try:
                        input_features = feature_extractor(audio_info['array'], sampling_rate=audio_info['sampling_rate']).input_features[0]
                        processed_data['input_features'].append(input_features)
                    except Exception as e:
                        print(f"Error processing audio: {e}")
                        processed_data['input_features'].append(None)
                else:
                    processed_data['input_features'].append(None)
        else:
            print("Batch does not contain 'audio' key or 'audio' key does not have the correct structure.")
            # Handling cases where 'audio' is not correctly formatted or missing
            processed_data['input_features'] = [None] * len(batch)  # Assuming batch size is reflected here
        return processed_data

    return dataset.map(prepare, batched=True, batch_size=50, num_proc=2)

def save_missing_files(missing_files):
    if missing_files:
        missing_files_df = pd.DataFrame(missing_files, columns=["missing_file_path"])
        missing_files_df.to_csv("../../data_processed/missing_audio_files.csv", index=False)
        print("Missing audio files saved to 'missing_audio_files.csv'.")
    else:
        print("No missing audio files detected.")

def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset with Whisper model.")
    parser.add_argument("model_size", type=str, choices=["small", "large"], help="Size of the Whisper model to use.")
    args = parser.parse_args()

    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{args.model_size}")
    csv_file_path = '../../data_processed/dataset_splitted.csv'
    df = pd.read_csv(csv_file_path)
    df['utterance_duration'] = (df['mark_end'] - df['mark_start']).astype(int)
    valid_speakers = df.groupby('name_unique_speaker')['utterance_duration'].sum()
    valid_speakers = valid_speakers[valid_speakers > 480000].index.tolist()

    df_filtered = df[df['name_unique_speaker'].isin(valid_speakers)]
    dataset = Dataset.from_pandas(df_filtered)
    processed_results = dataset.map(process_audio_file, batched=True, batch_size=50, num_proc=4)
    
    # Extract missing files from the results
    all_missing_files = [item for sublist in processed_results['missing_files'] for item in sublist if sublist]
    save_missing_files(all_missing_files)
    
    prepared_data = prepare_dataset(processed_results, feature_extractor)
    prepared_data.save_to_disk(f"../../data_processed/processed_dataset_{args.model_size}")
    print("Dataset processing completed and saved.")

if __name__ == "__main__":
    main()