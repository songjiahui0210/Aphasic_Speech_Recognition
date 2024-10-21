# based on https://huggingface.co/openai/whisper-large-v3

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pandas as pd
import warnings
import re 
import time

# suppress some warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# load dataset
csv_file_path = '../../data_processed/dataset_splitted.csv'
df = pd.read_csv(csv_file_path)

# get the test dataset
df = df[df['split'] == 'test']
num_rows = df.shape[0]
print(f"total rows: {num_rows}")
# get 100 rows
df = df.head(100)

# Check device and dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Add 'generated_transcriptions' column if it doesn't exist
if 'generated_transcriptions' not in df.columns:
    df['generated_transcriptions'] = ""


# path to the audio files
audio_base_path = "../../data_processed/audios/"
start_time = time.time()
for index, row in df.iterrows():
    print(f"Processing row{index+1}")
    audio_file_path = audio_base_path + row['folder_name'] + "/" + row['file_cut'] 

    try:
        result = pipe(audio_file_path, return_timestamps = True, generate_kwargs = {"language": "en"})
        generated_transcription = result["text"]
        df.at[index, 'generated_transcriptions'] = generated_transcription
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        df.at[index, 'generated_transcriptions'] = None

end_time = time.time()
# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time} seconds")

# Estimate total time based on the number of rows processed
estimated_total_time = (elapsed_time / 100) * num_rows/60/60
print(f"Estimated total time for full dataset: {estimated_total_time} hours")