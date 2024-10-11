import pandas as pd
import os

# Step 1: Load the dataset
data_path = '../data_processed/clean_dataset.csv'
df = pd.read_csv(data_path)

# Create a new folder for the splitted data
split_data_folder = '../data-splitted'
os.makedirs(split_data_folder, exist_ok=True)

# Step 2: Initialize a dictionary to store the splits
splits = {
    'train': [],
    'validation': [],
    'test': []
}

# Step 3: Process each category
categories = df['WAB_AQ_category'].unique()

for category in categories:
    category_data = df[df['WAB_AQ_category'] == category]
    
    # Shuffle the data
    category_data = category_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    n = len(category_data)
    train_end = int(0.8 * n)
    val_end = train_end + int(0.1 * n)
    
    # Append to the splits
    splits['train'].append(category_data.iloc[:train_end])
    splits['validation'].append(category_data.iloc[train_end:val_end])
    splits['test'].append(category_data.iloc[val_end:])

# Step 4: Concatenate and save the splits
train_df = pd.concat(splits['train'])
val_df = pd.concat(splits['validation'])
test_df = pd.concat(splits['test'])

# Save to new CSV files in the data-splitted folder
train_df.to_csv(os.path.join(split_data_folder, 'train_dataset.csv'), index=False)
val_df.to_csv(os.path.join(split_data_folder, 'validation_dataset.csv'), index=False)
test_df.to_csv(os.path.join(split_data_folder, 'test_dataset.csv'), index=False)

# Step 5: Move the corresponding audio files
def move_audio_files(df, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    for _, row in df.iterrows():
        audio_file = row['file']
        source_path = os.path.join('../data_processed/audios', audio_file)
        target_path = os.path.join(target_folder, audio_file)
        if os.path.exists(source_path):
            os.rename(source_path, target_path)

# Move audio files for each split into respective folders
move_audio_files(train_df, os.path.join(split_data_folder, 'train_audio_files'))
move_audio_files(val_df, os.path.join(split_data_folder, 'validation_audio_files'))
move_audio_files(test_df, os.path.join(split_data_folder, 'test_audio_files'))

