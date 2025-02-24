import argparse
from transformers import WhisperFeatureExtractor
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
import os 
import soundfile as sf

def process_audio_file(batch):
    audio_file_path = os.path.join("../../data_processed/audios", batch["folder_name"], batch["file_cut"])
    if os.path.exists(audio_file_path):
        audio, sample_rate = sf.read(audio_file_path)
        return {"audio": {"array": audio, "sampling_rate": sample_rate}}
    else:
        return {"audio": None}

def prepare_dataset(dataset, feature_extractor):
    def prepare(batch):
        audio = batch["audio"]
        if audio:
            input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            return {"input_features": input_features}
        else:
            return {"input_features": None}
    return dataset.map(prepare)

def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset with Whisper model.")
    parser.add_argument("model_size", type=str, choices=["small", "large-v3"],
                        help="Size of the Whisper model to use.")
    args = parser.parse_args()

    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{args.model_size}")
    csv_file_path = '../../data_processed/dataset_splitted.csv'
    df = pd.read_csv(csv_file_path)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(process_audio_file)
    dataset = prepare_dataset(dataset, feature_extractor)
    dataset.save_to_disk(f"../../data_processed/processed_dataset_{args.model_size}")

if __name__ == "__main__":
    main()