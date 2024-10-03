import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import evaluate # for WER calculation
import pandas as pd
import warnings
import re 
from generated_transcripts_processing import process_generated_transcriptions

# suppress some warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# load dataset
df = pd.read_csv('clean_dataset.csv')

df = df.head(10) # limit the first 10 rows for test

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

# Lists to hold all generated and ground truth transcriptions
all_generated_transcriptions = []
all_ground_truth_transcriptions = []

# path to the audio files
audio_base_path = "../data/audio_wav/English/Aphasia/Kempler/"
for index, row in df.iterrows():
    if "_nan" in row['file_cut']:
        continue  # skip those like kempler03a_nan_nan.wav

    audio_file_path = audio_base_path + row['file_cut'] 
    ground_truth_transcription = row['transcriptions']  

    # Generate transcription
    result = pipe(audio_file_path, return_timestamps=True, generate_kwargs={"language": "english"})

    # Append the generated and ground truth transcriptions to the lists
    generated_transcription = result["text"]
    print("Generated Transcription:", generated_transcription)
    print("ground_truth Transcription:", ground_truth_transcription)
    all_generated_transcriptions.append(generated_transcription)
    all_ground_truth_transcriptions.append(ground_truth_transcription)


processed_generated_transcriptions = process_generated_transcriptions(all_generated_transcriptions)


# initialize the WER metric
wer_metric = evaluate.load("wer")

# calculate overall WER
overall_error_rate = wer_metric.compute(predictions=processed_generated_transcriptions, references=all_ground_truth_transcriptions)
print("Overall Word Error Rate (WER):", overall_error_rate)