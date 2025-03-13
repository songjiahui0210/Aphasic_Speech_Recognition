import argparse
import pandas as pd
import os
import soundfile as sf
import numpy as np

from datasets import Dataset, load_from_disk
from transformers import WhisperProcessor

def process_audio_file(batch):

    audio_file_path = os.path.join("../../data_processed/audios", batch["folder_name"], batch["file_cut"])
    if os.path.exists(audio_file_path):
        audio, sample_rate = sf.read(audio_file_path)
        audio_list = audio.tolist() if isinstance(audio, np.ndarray) else audio
        return {
            "audio": {
                "array": audio_list,
                "sampling_rate": int(sample_rate)
            }
        }
    else:
        return {"audio": None}

def prepare_dataset(dataset, processor):
 

    def prepare(batch):
        processed_data = {
            "input_features": [],
            "labels": []
        }

        audio_list = batch.get("audio", [])
        if isinstance(audio_list, list):
            for audio_info in audio_list:
                if audio_info and "array" in audio_info and audio_info["array"] is not None:
                    try:
                        feats = processor.feature_extractor(
                            audio_info["array"],
                            sampling_rate=audio_info["sampling_rate"]
                        ).input_features[0]
                        processed_data["input_features"].append(feats)
                    except Exception as e:
                        print(f"Error processing audio: {e}")
                        processed_data["input_features"].append(None)
                else:
                    processed_data["input_features"].append(None)
        else:
            print("Batch does not contain a valid 'audio' field.")
            processed_data["input_features"] = [None] * len(batch["transcriptions"])

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

    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{args.model_size}")

    csv_file_path = '../../data_processed/set1_w_cohort.csv'
    df = pd.read_csv(csv_file_path)
    df = 

    # keep only speakers with total utterance_duration > 8 minutes
    df['utterance_duration'] = (df['mark_end'] - df['mark_start']).astype(int)
    speaker_durations = df.groupby('name_unique_speaker')['utterance_duration'].sum().reset_index()
    valid_speakers = speaker_durations[speaker_durations['utterance_duration'] > 480000]['name_unique_speaker']
    df_filtered = df[df['name_unique_speaker'].isin(valid_speakers)]

    dataset = Dataset.from_pandas(df_filtered)
    dataset = dataset.map(process_audio_file)
    dataset = prepare_dataset(dataset, processor)
    dataset.save_to_disk(f"../../data_processed/processed_dataset_{args.model_size}")

if __name__ == "__main__":
    main()