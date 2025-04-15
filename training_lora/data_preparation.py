import argparse
import os
import pandas as pd
import numpy as np
import soundfile as sf

from datasets import Dataset
from transformers import WhisperProcessor


def process_audio_file(batch):
    """
    For each record, construct the audio path from "folder_name" and "file_cut".
    If audio exists, read and store in batch["audio"] field as:
      batch["audio"] = {
          "array": [...],
          "sampling_rate": 16000
      }
    Otherwise, audio = None
    """
    audio_file_path = os.path.join("../../data_processed/audios", batch["folder_name"], batch["file_cut"])
    if os.path.exists(audio_file_path):
        audio, sample_rate = sf.read(audio_file_path)
        # If audio is numpy.ndarray, convert to list to avoid serialization issues
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
    For each sample in the dataset, use WhisperProcessor to generate:
      - batch["input_features"] (mel spectrogram)
      - batch["labels"] (token ids)

    Note: Processing one sample at a time (batched=False)
    For large datasets, consider changing to batched=True
    """

    def prepare(batch):
        try:
            audio_info = batch.get("audio", None)
            if audio_info and "array" in audio_info and audio_info["array"] is not None:
                # (1) Extract audio features using WhisperFeatureExtractor
                feats = processor.feature_extractor(
                    audio_info["array"],
                    sampling_rate=audio_info["sampling_rate"]
                ).input_features[0]
                batch["input_features"] = feats
            else:
                batch["input_features"] = None

            # (2) Text -> labels (token ids)
            text = batch.get("transcriptions", "")
            if isinstance(text, str) and len(text.strip()) > 0:
                tokenized = processor.tokenizer(text)
                batch["labels"] = tokenized["input_ids"]
            else:
                batch["labels"] = []
        except Exception as e:
            # Log errors and mark as None if processing fails
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

    # Apply prepare function to each sample
    dataset = dataset.map(prepare, batched=False, num_proc=1)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset with Whisper model.")
    parser.add_argument(
        "model_size",
        type=str,
        choices=["small","medium", "large"],
        help="Size of the Whisper model to use (e.g., 'small' or 'large' or 'medium')."
    )
    args = parser.parse_args()

    # 1) Load Processor (WhisperTokenizer + WhisperFeatureExtractor)
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{args.model_size}")

    # 2) Read CSV file
    csv_file_path = "../../data_processed/set2_validation.csv"
    df = pd.read_csv(csv_file_path)

    # 3) Filter speakers (example: keep only those with total duration > 480000 ms)
    df["utterance_duration"] = (df["mark_end"] - df["mark_start"]).astype(int)
    speaker_durations = df.groupby("name_unique_speaker")["utterance_duration"].sum().reset_index()
    valid_speakers = speaker_durations[speaker_durations["utterance_duration"] > 480000]["name_unique_speaker"]
    df_filtered = df[df["name_unique_speaker"].isin(valid_speakers)]
    print(f"Original CSV: {len(df)} rows, after filtering: {len(df_filtered)} rows.")

    # 4) Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df_filtered)

    # 5) Read audio into dataset["audio"]
    dataset = dataset.map(process_audio_file, batched=False)
    print(f"After attaching audio info, dataset size: {len(dataset)}")

    # 6) Generate input_features + labels using WhisperProcessor
    dataset = prepare_dataset(dataset, processor)
    print(f"After prepare_dataset, dataset size: {len(dataset)}")

    # 7) Filter out samples where input_features is None
    #    Avoid "All input_features are None" error during training
    dataset = dataset.filter(lambda x: (x.get("input_features") is not None) and (len(x["input_features"]) > 0))
    print(f"After filtering None input_features, dataset size: {len(dataset)}")

    # (Optional) Print first sample fields
    print("Sample 0:", dataset[0])

    # 8) Save to disk (includes input_features, labels, etc.)
    output_path = f"../../data_processed/processed_set2_validation_{args.model_size}"
    dataset.save_to_disk(output_path)

    print(f"All done! Dataset with 'input_features' and 'labels' is saved at: {output_path}")


if __name__ == "__main__":
    main()