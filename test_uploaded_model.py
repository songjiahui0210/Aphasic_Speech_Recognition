from transformers import pipeline
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# model_name="LitingZhou/whisper-small-vanilla" 
model_name="LitingZhou/whisper-large-freezing-5" # change this to the model to be tested

pipe = pipeline("automatic-speech-recognition", model=model_name)

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)

# Function to transcribe audio
def transcribe_audio(audio_path):
    # Load audio
    audio, sampling_rate = sf.read(audio_path)

    # Process input features
    input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features

    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription

audio_file_path = "../data_processed/audios/ACWT/ACWT01a_144.813_2.78.wav"
transcription = transcribe_audio(audio_file_path)
print("Transcription:", transcription)
