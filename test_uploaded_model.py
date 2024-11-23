# Use a pipeline as a high-level helper
from transformers import pipeline
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

pipe = pipeline("automatic-speech-recognition", model="LitingZhou/whisper-small-vanilla")

processor = AutoProcessor.from_pretrained("LitingZhou/whisper-small-vanilla")
model = AutoModelForSpeechSeq2Seq.from_pretrained("LitingZhou/whisper-small-vanilla")

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
