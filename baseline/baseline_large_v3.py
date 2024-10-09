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

df = df.head(1000) # limit the first 1000 rows for test

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

#-----------------------------------------------------------------

# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# import pandas as pd
# from datasets import Dataset, Audio
# import warnings

# # suppress some warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# # load dataset
# csv_file_path = '../../data_processed/clean_dataset.csv'
# df = pd.read_csv(csv_file_path)

# # df = df.head(10) # limit the first 10 rows for test

# # check device and dtype
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# # load the model and processor
# model_id = "openai/whisper-large-v3"
# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
# )
# model.to(device)

# processor = AutoProcessor.from_pretrained(model_id)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     torch_dtype=torch_dtype,
#     device=device,
# )

# # add 'generated_transcriptions_large' column if it doesn't exist
# if 'generated_transcriptions_large' not in df.columns:
#     df['generated_transcriptions_large'] = ""

# # path to the audio files
# audio_base_path = "../../data_processed/audios/"

# # create a dataset from the DataFrame
# def load_audio(row):
#     audio_file_path = audio_base_path + row['folder_name'] + "/" + row['file_cut']
#     return {"audio": audio_file_path}

# # create a Dataset object
# dataset = Dataset.from_pandas(df)

# # apply the function to add audio file paths
# dataset = dataset.map(load_audio)

# # add audio as a feature
# dataset = dataset.cast_column("audio", Audio())

# # generate transcriptions
# def transcribe(batch):
#     audio_files = [file["path"] for file in batch["audio"]]
#     # generate transcription for each audio file in the batch
#     results = pipe(audio_files, return_timestamps=False)
#     batch["generated_transcriptions_large"] = [result["text"] for result in results]
#     return batch

# # apply transcription in batches
# dataset = dataset.map(transcribe, batched=True, batch_size=8)

# # convert the dataset back to a DataFrame
# df = dataset.to_pandas()

# df.to_csv(csv_file_path, index=False)