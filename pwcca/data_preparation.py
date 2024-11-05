import argparse
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import soundfile as sf
from datasets import Dataset, DatasetDict, load_from_disk
import pandas as pd
import os
import numpy as np

def process_dataset(model_size):
    model_map = {
        "tiny": "openai/whisper-tiny",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large-v3"
    }

    if model_size not in model_map:
        raise ValueError("Model size must be one of 'tiny', 'small', 'medium', or 'large'.")

    model_name = model_map[model_size]

    # Load dataset
    csv_file_path = 'data_processed/dataset_splitted.csv'
    df = pd.read_csv(csv_file_path)

    columns_to_drop = ['mark_start', 'mark_end', 'name', 'sex', 'age', 'file', 'aphasia_type', 
                       'fluency_speech', 'original_file_length', 'difference', 
                       'name_extracted_from_filename']
    df = df.drop(columns=columns_to_drop)

    # Select one random audio per speaker in the test set only
    test_df = df[df['split'] == "train"]
    sampled_test = test_df.groupby("name_unique_speaker", group_keys=False).apply(lambda x: x.sample(1, random_state=42))

    print(f"Number of samples in the test set: {len(sampled_test)}")

    # Create DatasetDict with only the test set
    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(sampled_test)
    })

    print("Data splitting and sampling finished for test set only.")

    # Ensure output directory exists with CKA-specific naming
    processed_audio_data_path = f'data_processed/cka_processed_audio_dataset_{model_size}'
    missing_files = []

    def load_audio(batch):
        audio_file_path = os.path.join("data_processed/audios", batch["folder_name"], batch["file_cut"])
        if os.path.exists(audio_file_path):
            try:
                audio, sample_rate = sf.read(audio_file_path)
                batch["audio"] = {"array": audio, "sampling_rate": sample_rate}
            except Exception as e:
                print(f"Error loading {audio_file_path}: {e}")
                batch["audio"] = None
        else:
            print(f"File not found: {audio_file_path}")
            missing_files.append(audio_file_path)
            batch["audio"] = None
        return batch

    # Load or create processed dataset with audio
    if os.path.exists(processed_audio_data_path):
        dataset_dict = load_from_disk(processed_audio_data_path)
        print("Loaded existing audio dataset!")
    else:
        dataset_dict = dataset_dict.map(load_audio)
        dataset_dict.save_to_disk(processed_audio_data_path)
        print("Processed audio dataset saved.")

    # Save missing files to CSV if any
    if missing_files:
        missing_files_path = "data_processed/missing_audio_files.csv"
        pd.DataFrame(missing_files, columns=["missing_file_path"]).to_csv(missing_files_path, index=False)
        print(f"Missing audio files saved to '{missing_files_path}'.")

    # Initialize feature extractor and tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="English", task="transcribe")

    def prepare_dataset(batch):
        audio = batch["audio"]
        if audio is None or not audio["array"]:
            print(f"Invalid or empty audio data for batch {batch}")
            batch["input_features"], batch["labels"] = None, None
            return batch

        try:
            batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            batch["labels"] = tokenizer(batch["transcriptions"]).input_ids
        except Exception as e:
            print(f"Error processing audio: {e}")
            batch["input_features"], batch["labels"] = None, None
        return batch

    # Process dataset features for the test set only
    dataset_dict = dataset_dict.map(prepare_dataset, num_proc=6)
    print("Finished preparing dataset")

    # Save final processed dataset for CKA analysis
    dataset_dict_path = f'data_processed/cka_test_dataset_dict_{model_size}'
    os.makedirs(os.path.dirname(dataset_dict_path), exist_ok=True)
    dataset_dict.save_to_disk(dataset_dict_path)
    print("Test dataset_dict saved to disk for CKA analysis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a dataset with Whisper model.")
    parser.add_argument("model_size", type=str, choices=["tiny", "small", "medium", "large"],
                        help="Size of the Whisper model to use.")
    
    args = parser.parse_args()
    process_dataset(args.model_size)
