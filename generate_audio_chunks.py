# based on https://github.com/monirome/AphasiaBank/blob/main/audio_chunks.py

import os
import glob
import pandas as pd
import sys
from io import StringIO

def process_audio_chunks(transcript_folder_path, audio_folder_path):
    filepath = os.path.join(transcript_folder_path, 'clean_dataset.csv')

    if not os.path.exists(filepath):
        print(f"No clean_dataset.csv found in {transcript_folder_path}, skipping...")
        return

    df = pd.read_csv(filepath)

    for i in range(len(df)):
        file = glob.glob(os.path.join(audio_folder_path, f"""{df['file'][i]}"""), recursive=True)
        if not file:
            print(f"Audio file not found for {df['file'][i]}, skipping...")
            continue

        start = ((pd.to_numeric(df['mark_start'][i])) / 1000)
        duration = ((pd.to_numeric(df['mark_end'][i])) - (pd.to_numeric(df['mark_start'][i]))) / 1000
        output_file = f"{file[0][:-4]}_{start}_{duration}.wav"
        os.system(f"""sox {file[0]} {output_file} trim {start} {duration} 2>/dev/null""")

def process_all_folders(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            transcript_folder_path = os.path.join(root, dir_name)
            # print("transcript_folder_path", transcript_folder_path)
            audio_folder_path = transcript_folder_path.replace("transcripts", "audios")
            print(f"Processing folder: {audio_folder_path}")
            process_audio_chunks(transcript_folder_path, audio_folder_path)

################################################

transcripts_dir = "../data_processed/transcripts"
process_all_folders(transcripts_dir)