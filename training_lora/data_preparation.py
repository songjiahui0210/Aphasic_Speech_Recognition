import argparse
import pandas as pd
import os
import soundfile as sf
import numpy as np

from datasets import Dataset
from transformers import WhisperProcessor

def process_audio_file(batch):
    """
    batch["audio"].
    batch is dict, ex:
      {
        "folder_name": "...",
        "file_cut": "...",
        "transcriptions": "...",
        ...
      }
    """
    audio_file_path = os.path.join("../../data_processed/audios", batch["folder_name"], batch["file_cut"])
    if os.path.exists(audio_file_path):
        audio, sample_rate = sf.read(audio_file_path)
        audio_list = audio.tolist() if isinstance(audio, np.ndarray) else audio
        batch["audio"] = {
            "array": audio_list,
            "sampling_rate": int(sample_rate)
        }
    else:
        batch["audio"] = None
    return batch

def prepare_dataset(dataset, processor):
    """
      1)  WhisperFeatureExtractor  -> batch["input_features"]
      2)  tokenizer text to token IDs -> batch["labels"]
    """

    def prepare(batch):
        try:
            # (1)  input_features
            audio_info = batch.get("audio", None)
            if audio_info and "array" in audio_info and audio_info["array"] is not None:
                feats = processor.feature_extractor(
                    audio_info["array"],
                    sampling_rate=audio_info["sampling_rate"]
                ).input_features[0]
                batch["input_features"] = feats
            else:
                batch["input_features"] = None

            # (2)  labels
            text = batch.get("transcriptions", "")
            if isinstance(text, str):
                tokenized = processor.tokenizer(text)
                batch["labels"] = tokenized["input_ids"]
            else:
                batch["labels"] = []

        except Exception as e:
            # error
            error_message = (
                f"Error processing sample with folder_name={batch.get('folder_name')} "
                f"file_cut={batch.get('file_cut')} : {e}"
            )
            print(error_message)

            with open("error_report.txt", "a") as f:
                f.write(error_message + "\n")


            batch["input_features"] = None
            batch["labels"] = []

        return batch


    return dataset.map(prepare, batched=False, num_proc=1)

def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset with Whisper model.")
    parser.add_argument("model_size", type=str, choices=["small", "large"], help="Size of the Whisper model to use.")
    args = parser.parse_args()


    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{args.model_size}")


    csv_file_path = '../../data_processed/set1_w_cohort.csv'
    df = pd.read_csv(csv_file_path)


    df['utterance_duration'] = (df['mark_end'] - df['mark_start']).astype(int)
    speaker_durations = df.groupby('name_unique_speaker')['utterance_duration'].sum().reset_index()
    valid_speakers = speaker_durations[speaker_durations['utterance_duration'] > 480000]['name_unique_speaker']
    df_filtered = df[df['name_unique_speaker'].isin(valid_speakers)]


    dataset = Dataset.from_pandas(df_filtered)

    dataset = dataset.map(process_audio_file, batched=False)


    dataset = prepare_dataset(dataset, processor)


    dataset.save_to_disk(f"../../data_processed/processed_dataset_{args.model_size}_new")
    print("All done! Dataset with 'input_features' and 'labels' is saved.")

if __name__ == "__main__":
    main()
