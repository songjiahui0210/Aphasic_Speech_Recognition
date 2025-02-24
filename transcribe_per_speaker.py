import os
import pandas as pd
from transformers import pipeline
import torch

def transcribe_and_save_single_file(csv_path, audio_root, model_pipeline, detailed_csv):
    """
    Transcribe audio and save the predictions in a single CSV file, including all speakers.
    """
    # Load the dataset CSV
    df = pd.read_csv(csv_path)
    test_rows = df[df['split'] == 'test']

    # Ensure the output directory exists
    output_dir = os.path.dirname(detailed_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

    # Process each row in the test set
    first_write = True  
    with open(detailed_csv, mode='a') as f:
        for index, row in test_rows.iterrows():
            file_name = row['file_cut']
            transcription = row['transcriptions']
            folder_name = row['folder_name']
            audio_path = os.path.join(audio_root, folder_name, file_name)

            if not os.path.exists(audio_path):
                print(f"Audio file {audio_path} not found")
                continue

            # Transcribe audio using the pipeline
            print(f"Transcribing {audio_path}")
            result = model_pipeline(audio_path)
            predicted_text = result['text']

            # Collect data for detailed CSV
            output_data = {
                "speaker": row['name_unique_speaker'],
                "folder": folder_name,
                "file_name": file_name,
                "prediction": predicted_text,
                "reference": transcription
            }

            # Write data row-by-row to the CSV file
            output_df = pd.DataFrame([output_data])
            output_df.to_csv(f, mode='a', header=first_write, index=False)
            first_write = False

    print(f"Detailed results saved to {detailed_csv}")

def run_all_transcriptions_single_file(csv_path, audio_root, models, detailed_csv):
    """
    Run transcription for all models and save the results in a single file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name in models:
        torch.cuda.empty_cache() if device == "cuda" else None

        # Load the model pipeline
        model_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=0 if device == "cuda" else -1,  # 0 for GPU, -1 for CPU
            framework="pt",
        )

        # Transcribe and save results for this model to a single file
        transcribe_and_save_single_file(csv_path, audio_root, model_pipeline, detailed_csv)

        torch.cuda.empty_cache() if device == "cuda" else None

if __name__ == "__main__":
    csv_path = "../data_processed/dataset_splitted.csv"
    audio_root = "../data_processed/audios"
    models = ["openai/whisper-large"]
    detailed_csv = "../data_processed/detailed_wer_results_all_speakers.csv"

    run_all_transcriptions_single_file(csv_path, audio_root, models, detailed_csv)