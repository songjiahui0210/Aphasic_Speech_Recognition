import whisper
import pandas as pd
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation
import os
import inflect
import torch


def convert_numbers_to_words(text):
    """
    Convert digit numbers into words (e.g., 2002 -> two thousand and two).
    """
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

def calculate_wer(csv_path, audio_root, model, detailed_csv):
    """
    Calculate WER for the given model, process transcriptions, and save detailed results.
    """
    # Load the dataset CSV
    df = pd.read_csv(csv_path)
    test_rows = df[df['split'] == 'test']
    #limit_test_rows = test_rows.head(10)  # Limit to first 10 rows for testing

    # Open CSV in append mode
    with open(detailed_csv, mode='a') as f:
        first_write = True 

        # Process each row in the limited test set
        for index, row in test_rows.iterrows():
            file_name = row['file_cut']
            transcription = row['transcriptions']
            folder_name = row['folder_name']
            audio_path = os.path.join(audio_root, folder_name, file_name)

            if not os.path.exists(audio_path):
                print(f"Audio file {audio_path} not found")
                continue

            # Transcribe audio using Whisper
            print(f"Transcribing {audio_path}")
            result = model.transcribe(audio_path)
            predicted_text = result['text']

            # Normalize the prediction and transcription text
            transformation = Compose([ToLowerCase(), RemovePunctuation()])
            predicted_text = transformation(predicted_text)
            transcription = transformation(transcription)

            # Convert numbers in the predicted text to words
            predicted_text = convert_numbers_to_words(predicted_text)

            # Collect data for detailed CSV
            output_data = {
                "folder": folder_name,
                "file_name": file_name,
                "prediction": predicted_text,
                "reference": transcription,
                "wer": wer([transcription], [predicted_text])
            }

            # Write data row-by-row to the CSV file
            output_df = pd.DataFrame([output_data])
            output_df.to_csv(f, mode='a', header=first_write, index=False)
            first_write = False  

    print(f"Detailed results saved to {detailed_csv}")


def calculate_overall_wer_from_csv(detailed_csv):
    """
    Calculate the overall WER from the detailed CSV file after all predictions are made.
    """
    df = pd.read_csv(detailed_csv)
    references = df['reference'].tolist()
    predictions = df['prediction'].tolist()

    # Calculate overall WER using the list of references and predictions
    overall_wer = wer(references, predictions)
    return overall_wer


def run_all_models(csv_path, audio_root, models, detailed_results_folder, summary_csv):
    """
    Run WER calculations for all models and output summary and detailed results.
    """
    summary_data = []
    for model_size in models:
        # Avoid memory issues
        torch.cuda.empty_cache()

        model = whisper.load_model(model_size)

        detailed_csv = f"{detailed_results_folder}/detailed_{model_size}_results.csv"
        
        calculate_wer(csv_path, audio_root, model, detailed_csv)

        # Explicitly delete the model to free up memory
        del model
        torch.cuda.empty_cache()  # Free the memory after model is done

    #     # Calculate overall WER from the CSV
    #     overall_wer = calculate_overall_wer_from_csv(detailed_csv)
    #     print(f"Overall WER for {model_size}: {overall_wer}")

    #     summary_data.append({"model": model_size, "wer": overall_wer})

    # # Save summary WER results
    # summary_df = pd.DataFrame(summary_data)
    # summary_df.to_csv(summary_csv, index=False)
    # print(f"Summary WER results saved to {summary_csv}")

# Main script to run the models and calculations
csv_path = "../data_processed/dataset_splitted.csv"
audio_root = "../data_processed/audios"
models = ["base","small", "medium", "large", "large-v2", "large-v3"]
detailed_results_folder = "../data_processed/detailed_wer_results"
summary_csv = "../data_processed/summary_wer_results.csv"

if not os.path.exists(detailed_results_folder):
    os.makedirs(detailed_results_folder)
    print(f"Directory '{detailed_results_folder}' created.")

run_all_models(csv_path, audio_root, models, detailed_results_folder, summary_csv)
