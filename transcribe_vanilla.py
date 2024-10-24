import os
import warnings

# suppress some warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from transcribe import run_all_transcriptions

csv_path = "../data_processed/dataset_splitted.csv"
audio_root = "../data_processed/audios"

# use the fine-tuned model best checkpoint
models = ["../trained_models/whisper-small-vanilla"]
print("model loaded.")
detailed_results_folder = "../data_processed/detailed_wer_results"

if not os.path.exists(detailed_results_folder):
    os.makedirs(detailed_results_folder)
    print(f"Directory '{detailed_results_folder}' created.")

run_all_transcriptions(csv_path, audio_root, models, detailed_results_folder)