# based on https://huggingface.co/blog/fine-tune-whisper

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
import soundfile as sf
from datasets import Dataset, DatasetDict, Audio
import pandas as pd
import os 

# load dataset
csv_file_path = '../final_clean_dataset.csv'
df = pd.read_csv(csv_file_path)

def load_audio(batch):
    audio_file_path = os.path.join("../../data_processed/audios", batch["folder_name"], batch["file_cut"])
    audio, _ = sf.read(audio_file_path)
    batch["audio"] = {"array": audio}
    return batch

dataset = Dataset.from_pandas(df)
# load_audio(dataset[0])
# map the function to load audio files
dataset = dataset.map(load_audio)


# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

# dataset = Dataset.from_pandas(df)
# print(dataset[0])

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch
