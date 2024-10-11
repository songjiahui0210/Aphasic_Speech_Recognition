# Data processing

After git clone the repo, enter the directory
```
cd Aphasic_speech_recognition
```
### Step 1: unzip the transcripts

```
chmod +x open_zip.sh
./open_zip.sh
```
Then, manually change the name of the second folder "Adler" to "adler", make it consistent.

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
