import whisper
import pandas as pd
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation
import os
import inflect
import torch


def convert_numbers_to_words(text):
    p = inflect.engine()
    words = text.split()
    converted_text = ""
    for word in words:
        if word.isdigit():
            try:
                word = p.number_to_words(word)
            except inflect.NumOutOfRangeError:
                pass
        converted_text += word + " "
    return converted_text.strip()


def calculate_wer(csv_path, audio_root, model, detailed_csv, split_name):
    df = pd.read_csv(csv_path)
    selected_rows = df[df['split'].str.lower() == split_name.lower()]

    with open(detailed_csv, mode='a') as f:
        first_write = True

        for index, row in selected_rows.iterrows():
            file_name = row['file_cut']
            transcription = row['transcriptions']
            folder_name = row['folder_name']
            audio_path = os.path.join(audio_root, folder_name, file_name)

            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                continue

            print(f"Transcribing {audio_path}")
            result = model.transcribe(audio_path)
            predicted_text = result['text']

            transformation = Compose([ToLowerCase(), RemovePunctuation()])
            predicted_text = transformation(predicted_text)
            transcription = transformation(transcription)
            predicted_text = convert_numbers_to_words(predicted_text)

            output_data = {
                "folder": folder_name,
                "file_name": file_name,
                "prediction": predicted_text,
                "reference": transcription,
                "wer": wer([transcription], [predicted_text])
            }

            output_df = pd.DataFrame([output_data])
            output_df.to_csv(f, mode='a', header=first_write, index=False)
            first_write = False

    print(f"Detailed results saved to {detailed_csv}")


def run_all_models(csv_path, audio_root, models, detailed_results_folder, summary_csv, splits):
    summary_data = []
    for model_size in models:
        torch.cuda.empty_cache()
        model = whisper.load_model(model_size)

        for split in splits:
            detailed_csv = os.path.join(detailed_results_folder, f"detailed_{model_size}_{split}_results.csv")
            print(f"Running model: {model_size}, split: {split}")
            calculate_wer(csv_path, audio_root, model, detailed_csv, split_name=split)

        del model
        torch.cuda.empty_cache()


# === Main Execution ===
csv_path = "../data_processed/dataset_splitted_by_duration.csv"
audio_root = "../data_processed/audios"
models = ["base"]
splits = ["train", "validation", "test"]
detailed_results_folder = "../data_processed/detailed_wer_results"
summary_csv = "../data_processed/summary_wer_results.csv"

if not os.path.exists(detailed_results_folder):
    os.makedirs(detailed_results_folder)
    print(f"📁 Created result folder: '{detailed_results_folder}'")

run_all_models(csv_path, audio_root, models, detailed_results_folder, summary_csv, splits)
