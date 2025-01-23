# example usgae: python3 wer_calculation_finetune.py "../data_processed/detailed_wer_results/detailed_.._trained_models_whisper-small-vanilla_results.csv"

import sys
from wer_calculation import calculate_overall_wer_from_csv

if len(sys.argv) != 2:
    print("Usage: python3 wer_calculation_finetune.py <detailed_wer_csv_path>")
    sys.exit(1)

detailed_csv = sys.argv[1]

overall_wer = calculate_overall_wer_from_csv(detailed_csv)

print(f"Overall WER for vanilla fine-tuned small model: {overall_wer}")