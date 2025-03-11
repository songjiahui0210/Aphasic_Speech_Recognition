import pandas as pd
import os
from sklearn.model_selection import train_test_split


dataset_path = "../../data_processed/dataset_splitted.csv"
df = pd.read_csv(dataset_path)


speaker_durations = df.groupby("name_unique_speaker").apply(
    lambda x: (x["mark_end"] - x["mark_start"]).sum()
)

# > 8min= 480000ms
valid_speakers = speaker_durations[speaker_durations > 480000].index.tolist()
filtered_df = df[df["name_unique_speaker"].isin(valid_speakers)]

# split set 1 and 2(70%,30%)
speakers_set1, speakers_set2 = train_test_split(
    valid_speakers, test_size=0.3, random_state=42
)

set1_df = filtered_df[filtered_df["name_unique_speaker"].isin(speakers_set1)]
set2_df = filtered_df[filtered_df["name_unique_speaker"].isin(speakers_set2)]

# Set 2: 80% Train, 10% Validation, 10% Test
speakers_set2_enroll, speakers_set2_temp = train_test_split(speakers_set2, test_size=0.2, random_state=42)
speakers_set2_val, speakers_set2_test = train_test_split(speakers_set2_temp, test_size=0.5, random_state=42)

# Data of Enrollment, Validation, Test
set2_enroll_df = set2_df[set2_df["name_unique_speaker"].isin(speakers_set2_enroll)]
set2_val_df = set2_df[set2_df["name_unique_speaker"].isin(speakers_set2_val)]
set2_test_df = set2_df[set2_df["name_unique_speaker"].isin(speakers_set2_test)]


output_dir = "../../data_processed/"
os.makedirs(output_dir, exist_ok=True)

set1_df.to_csv(os.path.join(output_dir, "set1_w_cohort.csv"), index=False)
set2_enroll_df.to_csv(os.path.join(output_dir, "set2_enrollment.csv"), index=False)
set2_val_df.to_csv(os.path.join(output_dir, "set2_validation.csv"), index=False)
set2_test_df.to_csv(os.path.join(output_dir, "set2_test.csv"), index=False)


print(f"Data splitting have complete!\n"
      f"- Set 1 (W_cohort) specker: {len(speakers_set1)}\n"
      f"- Set 2 (personalize adapter) specker: {len(speakers_set2)}\n"
      f"  - Enrollment specker (Train 80%): {len(speakers_set2_enroll)}\n"
      f"  - Validation specker (Valid 10%): {len(speakers_set2_val)}\n"
      f"  - Test specker (Test 10%): {len(speakers_set2_test)}\n")
