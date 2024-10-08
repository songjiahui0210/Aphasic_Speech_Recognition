import evaluate # for WER calculation
import pandas as pd
from generated_transcripts_processing import process_generated_transcriptions

# load transcripts
columns_to_read = {'generated_transcriptions_large':'generated_transcriptions', 'transcriptions':'reference_transcriptions'}
df = pd.read_csv('../../data_processed/clean_dataset.csv', usecols=columns_to_read.keys())
df = df.rename(columns=columns_to_read)

df['processed_generated_transcriptions'] = process_generated_transcriptions(df['generated_transcriptions'])
df.to_csv('generated_transcriptions_large.csv', index=False) 

# initialize the WER metric
wer_metric = evaluate.load("wer")

# calculate overall WER
overall_error_rate = wer_metric.compute(predictions = df['processed_generated_transcriptions'], references=df['reference_transcriptions'] )
print("Overall Word Error Rate (WER):", overall_error_rate)