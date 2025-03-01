import pandas as pd
import os

# pre: Check if the output file already exists and delete it if it does
output_path = '../data_processed/dataset_splitted_by_duration.csv'
if os.path.exists(output_path):
    os.remove(output_path)

# Step 1: Load the dataset
data_path = 'final_clean_dataset.csv'
df = pd.read_csv(data_path)

# Step 2: Calculate duration for each row
df['duration'] = (df['mark_end'] - df['mark_start']) / 1000  # convert to seconds

# Step 3: Calculate total duration per name_unique_speaker
duration_per_speaker = df.groupby('name_unique_speaker')['duration'].sum().reset_index()

# Step 4: Identify speakers with total duration > 8 minutes (480 seconds)
long_duration_speakers = duration_per_speaker[duration_per_speaker['duration'] > 480]['name_unique_speaker'].tolist()

# Step 5: Mark 'train' for long duration speakers
df['split'] = df['name_unique_speaker'].apply(lambda x: 'train' if x in long_duration_speakers else None)

# Step 6: Split remaining speakers into 'test' and 'eval' by speaker
remaining_speakers = duration_per_speaker[~duration_per_speaker['name_unique_speaker'].isin(long_duration_speakers)]['name_unique_speaker'].tolist()

# Shuffle speakers randomly for unbiased split
remaining_speakers = pd.Series(remaining_speakers).sample(frac=1, random_state=42).tolist()

# Split 50% of remaining speakers into 'test' and 'eval'
mid_point = len(remaining_speakers) // 2

test_speakers = remaining_speakers[:mid_point]
eval_speakers = remaining_speakers[mid_point:]

# Assign the splits to the dataframe
df.loc[df['name_unique_speaker'].isin(test_speakers), 'split'] = 'test'
df.loc[df['name_unique_speaker'].isin(eval_speakers), 'split'] = 'eval'

# Step 7: Save the updated DataFrame to a new CSV file
df.to_csv(output_path, index=False)

print(f"Data splitting completed and saved to '{output_path}'!")
