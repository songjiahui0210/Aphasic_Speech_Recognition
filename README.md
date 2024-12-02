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

### Step 7: calculating baseline WER
```
python3 transcribe.py
python3 wer_calculation.py
```

# Fine-tuning

```
cd Aphasic_speech_recognition/vanilla_training/
```
### Step 1: prepare data

Prepare data, including loading audio files, compute log-Mel input features, and encode transcriptions to label ids.
```
python3 data_preparation.py <model_size>
```
Select model size from "tiny", "small", "medium" and "large".

### Step 2: train the model

Vanilla fine-tuning, select model size from "small", "medium" and "large":
```
python3 training.py <model_size>
```

Fine-tuning with encoder freezing, select model size from "small", "medium" and "large" and set the number of encoder layers to freeze:
```
python3 training.py <model_size> --freeze_layers <number_of_encoder_layers_to_freeze>
```

### Step 3: evaluation

After training, run the following commands step by step to get WER result.
```
cd ..
python3 transcribe_finetune.py <model_path>
python3 wer_calculation_finetune.py <detailed_wer_csv_path>
```

# CKA Analysis
Brefore cka_analysis, if the dataset is in a subdirectory or parent directory of the cka_analysis.py, adjust the path accordingly.
```
cd cka
python3 cka_analysis.py
```


