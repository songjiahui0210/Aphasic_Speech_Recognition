import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import pandas as pd
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation
import os
import inflect
import librosa
import numpy as np


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

def calculate_wer(csv_path, audio_root, model, processor, detailed_csv, device):
    """
    Calculate WER for the given model and dataset, process transcriptions, and save detailed results.
    """
    # Load the dataset CSV
    df = pd.read_csv(csv_path)
    
    # Open CSV in append mode
    with open(detailed_csv, mode='a') as f:
        first_write = True 

        # Process each row in the dataset
        for index, row in df.iterrows():
            file_name = row['file_cut']
            transcription = row['transcriptions']
            folder_name = row['folder_name']
            audio_root = "../../data_processed/audios"
            audio_path = os.path.join(audio_root, folder_name, file_name)

            if not os.path.exists(audio_path):
                print(f"Audio file {audio_path} not found")
                continue

            # Transcribe audio using Whisper via transformers
            print(f"Transcribing {audio_path}")
            
            try:
                # Load audio with librosa
                audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
                
                # Process audio with Whisper processor
                input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
                
                # Generate token ids
                with torch.no_grad():
                    predicted_ids = model.generate(input_features.to(device))
                
                # Decode token ids to text
                predicted_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

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
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue

    print(f"Detailed results saved to {detailed_csv}")


def calculate_overall_wer_from_csv(detailed_csv):
    """
    Calculate the overall WER from the detailed CSV file after all predictions are made.
    """
    df = pd.read_csv(detailed_csv)
    references = [str(ref) if not pd.isna(ref) else "" for ref in df["reference"]]
    predictions = [str(pred) if not pd.isna(pred) else "" for pred in df["prediction"]]

    # Calculate overall WER using the list of references and predictions
    overall_wer = wer(references, predictions)
    return overall_wer


def run_baseline_evaluation(model_size, validation_csv, test_csv, audio_root, results_folder):
    """
    Run baseline WER calculations for validation and test sets.
    """
    # Avoid memory issues
    torch.cuda.empty_cache()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model_id = f"openai/whisper-{model_size}"
    print(f"Loading {model_id} model...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    processor = WhisperProcessor.from_pretrained(model_id, language="en", task="transcribe")
    
    # Process validation set
    validation_output_csv = f"{results_folder}/baseline_{model_size}_validation_results.csv"
    print(f"Processing validation set...")
    calculate_wer(validation_csv, audio_root, model, processor, validation_output_csv, device)
    validation_wer = calculate_overall_wer_from_csv(validation_output_csv)
    print(f"Validation WER for {model_size}: {validation_wer}")
    
    # Process test set
    test_output_csv = f"{results_folder}/baseline_{model_size}_test_results.csv"
    print(f"Processing test set...")
    calculate_wer(test_csv, audio_root, model, processor, test_output_csv, device)
    test_wer = calculate_overall_wer_from_csv(test_output_csv)
    print(f"Test WER for {model_size}: {test_wer}")
    
    # Save summary results
    summary_data = {
        "model": model_size,
        "validation_wer": validation_wer,
        "test_wer": test_wer
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_csv = f"{results_folder}/baseline_{model_size}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary results saved to {summary_csv}")
    
    # Explicitly delete the model to free up memory
    del model
    torch.cuda.empty_cache()
    
    return validation_wer, test_wer

# Main script settings
model_size = "medium"  # Change this to the model size that want to evaluate
validation_csv = "../../data_processed/set2_validation.csv"
test_csv = "../../data_processed/set2_test.csv"
audio_root = "../../data_processed/audios"
results_folder = "../../data_processed/baseline_wer_results"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    print(f"Directory '{results_folder}' created.")

# Run the evaluation
validation_wer, test_wer = run_baseline_evaluation(
    model_size, 
    validation_csv, 
    test_csv, 
    audio_root, 
    results_folder
)

print(f"\nFinal Results for {model_size} model:")
print(f"Validation WER: {validation_wer}")
print(f"Test WER: {test_wer}")