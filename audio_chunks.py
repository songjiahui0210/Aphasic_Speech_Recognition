# learn from https://github.com/monirome/AphasiaBank/blob/main/audio_chunks.py


filepath = "clean_dataset.csv"
audiopath = "../data/audio_wav/English/Aphasia/Kempler" 

################################################
import os
import glob
import pandas as pd
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
################################################   
# Get Audio Chunks                                                                         

df = pd.read_csv(filepath)
print(f"Number of samples: {len(df)}")  

for i in range(len(df)):
    file = glob.glob(audiopath + f"""/{df['file'][i]}""", recursive=True)
    print(file[0])
    start=((pd.to_numeric(df['mark_start'][i]))/1000)
    print(start)
    end=((pd.to_numeric(df['mark_end'][i]))-(pd.to_numeric(df['mark_start'][i])))/1000
    os.system(f"""sox {file[0]} {file[0][:-4]}_{start}_{end}.wav trim {start} {end}""")