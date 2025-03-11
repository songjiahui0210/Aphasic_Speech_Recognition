import argparse
from transformers import WhisperFeatureExtractor
import pandas as pd
from datasets import Dataset, load_from_disk
import os
import soundfile as sf
import numpy as np

def process_audio_file(batch):
    audio_file_path = os.path.join("../../data_processed/audios", batch["folder_name"], batch["file_cut"])
    if os.path.exists(audio_file_path):
        audio, sample_rate = sf.read(audio_file_path)
        audio_list = audio.tolist() if isinstance(audio, np.ndarray) else audio
        return {"audio": {"array": audio_list, "sampling_rate": int(sample_rate)}}
    else:
        return {"audio": None}


def prepare_dataset(dataset, feature_extractor):
    def prepare(batch):
        processed_data = {'input_features': [], 'labels': [], 'split':batch['split'] }
        # Ensure that 'audio' is in batch and is iterable
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
            processed_data['input_features'] = [None] * len(batch)

        
        # ensure label exit
        if 'text_transcription' in batch:
            processed_data['labels'] = batch['text_transcription']
        else:
            raise KeyError("Dataset is missing 'text_transcription' for labels.")
  
        return processed_data

    return dataset.map(prepare, batched=True, batch_size=50)

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
    csv_file_path = '../../data_processed/set1_w_cohort.csv'
    df = pd.read_csv(csv_file_path)

    print(f"CSV Columns: {list(df.columns)}")

    # use 'transcriptions' as text_transcription
    if 'transcriptions' not in df.columns:
        raise KeyError(f"Expected 'transcriptions' in dataset, but found: {list(df.columns)}")
    if 'split' not in df.columns:
        raise KeyError(f"Expected 'split' in dataset, but found: {list(df.columns)}")


    df['utterance_duration'] = (df['mark_end'] - df['mark_start']).astype(int)
    
    # Sum durations by speaker or file
    speaker_durations = df.groupby('name_unique_speaker')['utterance_duration'].sum().reset_index()

    # Filter speakers by total duration > 480,000 milliseconds (8 minutes)
    valid_speakers = speaker_durations[speaker_durations['utterance_duration'] > 480000]['name_unique_speaker']
    
    # Filter the original dataframe to include only valid speakers
    df_filtered = df[df['name_unique_speaker'].isin(valid_speakers)]

    # ensure df_filtered include transcriptions
    df_filtered = df_filtered[['folder_name', 'file_cut', 'transcriptions', 'split']].copy()
    df_filtered.rename(columns={'transcriptions': 'text_transcription'}, inplace=True)

    dataset = Dataset.from_pandas(df_filtered)
    dataset = dataset.map(process_audio_file)
    dataset = prepare_dataset(dataset, feature_extractor)
    dataset.save_to_disk(f"../../data_processed/processed_dataset_{args.model_size}")

if __name__ == "__main__":
    main()