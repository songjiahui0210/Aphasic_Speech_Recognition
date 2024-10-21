# based on https://huggingface.co/openai/whisper-large-v3
# usage example: python3 baseline.py "small"

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pandas as pd
import warnings
import re
import argparse

# suppress some warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def run_baseline(model_size):
    # map model size to the corresponding model name
    model_map = {
        "tiny": "openai/whisper-tiny",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large-v3"
    }

    if model_size not in model_map:
        raise ValueError("Model size must be one of 'tiny', 'small', 'medium', or 'large'.")
    
    model_id = model_map[model_size]

    # load dataset
    csv_file_path = '../../data_processed/dataset_splitted.csv'
    df = pd.read_csv(csv_file_path)

    # get the test dataset
    df = df[df['split'] == 'test']

    # Check device and dtype
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # load model
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

    # add 'generated_transcriptions' column if it doesn't exist
    if 'generated_transcriptions' not in df.columns:
        df['generated_transcriptions'] = ""


    # path to the audio files
    audio_base_path = "../../data_processed/audios/"
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

    df = df.rename(columns={'transcriptions': 'reference_transcriptions'})
    output_df = df[['reference_transcriptions', 'generated_transcriptions']]
    output_file = f"generated_transcriptions_{model_size}.csv"
    output_df.to_csv(output_file, index=False)
    print(f"Transcriptions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run transcription using different Whisper model sizes.")
    parser.add_argument("model_size", type=str, choices=["tiny", "small", "medium", "large"],
                        help="Size of the Whisper model to use.")
    
    args = parser.parse_args()

    run_baseline(args.model_size)