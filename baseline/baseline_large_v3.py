import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pandas as pd
import warnings
import re 

# suppress some warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# load dataset
csv_file_path = '../../data_processed/clean_dataset.csv'
df = pd.read_csv(csv_file_path)

# df = df.head(10) # limit the first 10 rows for test

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

# Add 'generated_transcriptions_large' column if it doesn't exist
if 'generated_transcriptions_large' not in df.columns:
    df['generated_transcriptions_large'] = ""


# path to the audio files
audio_base_path = "../../data_processed/audios/"
for index, row in df.iterrows():
    print(f"Processing row{index+1}")
    audio_file_path = audio_base_path + row['folder_name'] + "/" + row['file_cut'] 

    # Skip if 'generated_transcriptions_large' is not empty
    if row['generated_transcriptions_large'].strip() != "":
        continue

    # Generate transcription
    try:
        result = pipe(audio_file_path, return_timestamps = True, generate_kwargs = {"language": "english"})
        generated_transcription = result["text"]
        df.at[index, 'generated_transcriptions_large'] = generated_transcription
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        df.at[index, 'generated_transcriptions_large'] = None

df.to_csv(csv_file_path, index=False)

