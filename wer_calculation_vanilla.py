# from jiwer import wer, Compose, ToLowerCase, RemovePunctuation
# import pandas as pd
# import inflect
from wer_calculation import calculate_overall_wer_from_csv


models = ["openai/whisper-small_vanilla"]
detailed_csv = "../data_processed/detailed_wer_results/detailed_openai_whisper-small_results.csv"
summary_csv = "../data_processed/summary_wer_results.csv"

overall_wer = calculate_overall_wer_from_csv(detailed_csv)

print(f"Overall WER for vanilla fine-tuned small model: {overall_wer}")