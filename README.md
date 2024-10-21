# Data processing

After git clone the repo, enter the directory
```
cd Aphasic_speech_recognition/
```

```
module load python/3.8.1
```
### Step 1: unzip the transcripts

```
chmod +x open_zip.sh
./open_zip.sh
```
Then, go to /data_processed/transcripts/, manually change the name of the second folder "Adler" to "adler", make it consistent.

Before Step 2

```
pip install pylangacq
```

Upgrade to python-dateutil-2.9.0

```
pip install --upgrade python-dateutil --user
```
### Step 2: process the transcripts

```
chmod +x data_processing.sh
./data_processing.sh
```

### Step 3: convert audio to .wav
```
chmod +x convert_to_wav.sh
./convert_to_wav.sh
```
### Step 4: generate audio chunks

```
python3 generate_audio_chunks.py
```

Note: Steps 2, 3 and 4 are very time-consuming.

### Step 5: check data statistics and do more data cleaning
Delete the rows with empty transcriptions, audios longer than 30 seconds, and audios shorter than 0.3 seconds.
```
python3 data_cleaning_and_statistics.py
```

### Step 6: split data 
Split the dataset into training (80%), validation (10%), and test (10%) sets based on the unique speakers within each WAB_AQ_category

```
python3 data_splitting.py
```

# Baseline

### Step 7: Calculating baseline WER
```
python3 transcribe.py
python3 wer_calculation.py
```

# Vanilla fine-tuning
```
cd Aphasic_speech_recognition/vanilla_training/
```
### step 1: Prepare data
prepare data, including loading audio files, compute log-Mel input features, and encode transcriptions to label ids
```
python3 data_preparation.py "small"
```
Select model size from "tiny", "small", "medium" and "large".
### step 2: train the model
```
python3 training.py "small"
```
Select model size from "tiny", "small", "medium" and "large".
