import jiwer # for WER calculation
import pandas as pd
from generated_transcripts_processing import process_generated_transcriptions

# load transcripts
columns_to_read = {'generated_transcriptions_large':'generated_transcriptions', 'transcriptions':'reference_transcriptions'}
df = pd.read_csv('../../data_processed/clean_dataset.csv', usecols=columns_to_read.keys())
df = df.rename(columns=columns_to_read)

df['processed_generated_transcriptions'] = process_generated_transcriptions(df['generated_transcriptions'])

row_wers = []
transformed_predictions =[]
transformed_references = []

# initialize the WER metric
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),           
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

for index, row in df.iterrows():
    prediction = transformation(row['processed_generated_transcriptions'])
    reference = transformation(row['reference_transcriptions'])
    transformed_predictions.append(prediction)
    transformed_references.append(reference)

    try:
        wer = jiwer.wer(reference, prediction)
        row_wers.append(wer)
    except ValueError as e:
        print(f"Error at row {index}: {e}")
        row_wers.append(None)

df['transformed_predictions'] = transformed_predictions
df['transformed_references'] = transformed_references
df['wer'] = row_wers
df.to_csv('generated_transcriptions_large.csv', index=False)

# calculate overall WER
overall_error_rate = jiwer.wer(transformed_references, transformed_predictions)
print("Overall Word Error Rate (WER):", overall_error_rate)