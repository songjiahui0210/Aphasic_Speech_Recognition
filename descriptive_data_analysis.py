import pandas as pd

data_path = '../data_processed/dataset_splitted.csv'
df = pd.read_csv(data_path)
df['duration_seconds'] = (df['mark_end'] - df['mark_start']) / 1000

speaker_duration_total = df.groupby('name_extracted_from_filename')['duration_seconds'].sum()

# Calculate descriptive statistics for the summed durations
# stats = speaker_duration_total.describe()
# print(stats)

long_duration_speakers = speaker_duration_total[speaker_duration_total > 480] # 413/634
num_speakers_over_eight_minutes = long_duration_speakers.count()
print(f"Number of speakers with total duration over 8 minutes: {num_speakers_over_eight_minutes}")

# count     634.000000
# mean      785.676055
# std       776.196778
# min         0.300000
# 25%       381.315500
# 50%       618.132500
# 75%       882.920000
# max      7327.261000