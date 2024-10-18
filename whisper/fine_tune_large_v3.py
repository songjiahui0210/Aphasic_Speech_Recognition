# based on https://huggingface.co/blog/fine-tune-whisper

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
import soundfile as sf
from datasets import Dataset, DatasetDict, Audio
import pandas as pd
import os 

# load dataset
csv_file_path = '../../data_processed/dataset_splitted.csv'
df = pd.read_csv(csv_file_path)

# directory to save the processed dataset
processed_data_path = '../../data_processed/processed_audio_dataset'
final_processed_data_path = '../../data_processed/final_processed_dataset'

def load_audio(batch):
    audio_file_path = os.path.join("../../data_processed/audios", batch["folder_name"], batch["file_cut"])
    audio, _ = sf.read(audio_file_path)
    batch["audio"] = {"array": audio}
    return batch

# check if the processed dataset already exists
if os.path.exists(processed_data_path):
    # load preprocessed dataset
    dataset = Dataset.load_from_disk(processed_data_path)
    print("Loaded dataset from disk!")
else:
    # create dataset from the CSV and map the function to load audio files
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(load_audio)
    
    # save the processed dataset for future use
    dataset.save_to_disk(processed_data_path)
    print("Processed dataset saved to disk!")

print(dataset[0])

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]
    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcriptions"]).input_ids
    return batch

if os.path.exists(final_processed_data_path):
    # load fully preprocessed dataset
    dataset = Dataset.load_from_disk(final_processed_data_path)
    print("Loaded final processed dataset from disk!")
else:
    # prepare the dataset
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=8)
    
    # save the fully processed dataset for future use
    dataset.save_to_disk(final_processed_data_path)
    print("Final processed dataset saved to disk!")

print("after prepare_data", dataset[0])