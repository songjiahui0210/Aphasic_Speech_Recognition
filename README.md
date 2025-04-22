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
nohup ./data_processing.sh > output.log 2>&1 &
```

### Step 3: convert audio to .wav

```
chmod +x convert_to_wav.sh
nohup ./convert_to_wav.sh > output.log 2>&1 &
```

### Step 4: generate audio chunks

```
nohup python3 generate_audio_chunks.py > output.log 2>&1 &
```

### Step 5: check data statistics and do more data cleaning

Delete the rows with empty transcriptions, audios longer than 30 seconds, and audios shorter than 0.3 seconds.

```
nohup python3 data_cleaning_and_statistics.py > output.log 2>&1 &
```

### Step 6: split data

We split the full dataset into two main sets and then further partition Set 2 for enrollment, validation, and testing.
**Divide speakers into Set 1 and Set 2**  
   - Randomly assign 70 % of valid speakers to **Set 1** (cohort-based adaptation).  
   - Assign remaining 30 % to **Set 2** (personalized-adapter experiments).
**Further split Set 2**  
   - **Enrollment (train)**: 80 % of Set 2 speakers. 
     - **Validation** (10 % of original Set 2)  
     - **Test** (10 % of original Set 2)

```
nohup python3 training_lora/data_splitting_lora.py
```

# Baseline

### Calculating baseline WER for whole dataset, and WER of Set 1, Set 2(validation, test:
```
cd ..
python3 transcribe.py
python3 wer_calculation.py
```
### Calculating baseline WER of Set 1, Set 2(validation, test:
```
python3 training_lora/baseline_wer.py
```


# Fine-tuning in Set 1: cohort training 

```
cd Aphasic_speech_recognition/training_lora/
```

### Step 1: prepare data for set 1

Generate log‑Mel inputs and tokenized transcripts for cohort training. 

```
python3 data_preparation.py <model_size>
```

Select model size from "small" and "medium".

### Step 2: split & filter Set 1 (W_cohort)

```
python3 filter.py
python3 filter_process_dataset.py
```

**Fine-tuning in Set 1: will get the model after fine-tuning

```
python3 train_lora.py 
```
# Fine-tuning in Set 2: Personalized Adaptation

### Step 3: enrollment, validation and test

After training in set 1, run the following commands step by step to get the WER of enrollment, validation, and test results.

```
python3 tran_set2_lora.py
```
After running the enrollment, save the model path and change in the tran_set2_validation.py and tran_set2_test.py:

```
python3 tran_set2_validation.py
python3 tran_set2_test.py
```

# Step 4: Parameter Adjustment

Before runing, adjust the path of model path to the model save after step 2, and change the rank r and alpha a.
```
python3 train_enrollment.py
```
After running the enrollment, save the model path and change in the train_validation.py and train_test.py:
```
python3 train_validation.py
python3 train_test.py
```


# Model & Data

- **Whisper‑small LoRA (r=8, α=16)**  
  https://huggingface.co/liulian26/aphasic-whisper-small-lora
  
