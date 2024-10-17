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

def load_audio(batch):
    audio_file_path = os.path.join("../../data_processed/audios", batch["folder_name"], batch["file_cut"])
    audio, _ = sf.read(audio_file_path)
    batch["audio"] = {"array": audio}
    return batch

dataset = Dataset.from_pandas(df)
# map the function to load audio files
dataset = dataset.map(load_audio)
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

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)

