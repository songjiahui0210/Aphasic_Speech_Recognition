import os
import soundfile as sf
from pydub import AudioSegment
import wave

# file_path="../../data_processed/audios/BU/BU12a_1513.241_3.84.wav"
file_path="../../data_processed/audios/BU/BU12a.wav"

file_size = os.path.getsize(file_path)
print(f"File size: {file_size} bytes")
if file_size == 0:
    print("File is empty.")

# try:
#     audio, sample_rate = sf.read(file_path)
#     print("Audio loaded successfully.")
#     print(f"Audio shape: {audio.shape}, Sample Rate: {sample_rate}")
# except Exception as e:
#     print(f"Error loading audio file: {e}")


# # play audio
# try:
#     audio_segment = AudioSegment.from_file(file_path)
#     print("Audio file is playable.")
#     # audio_segment.play() 
# except Exception as e:
#     print(f"Error playing audio file: {e}")


def get_wav_length(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()  # Sample rate
        num_frames = wav_file.getnframes()  # Total number of frames
        duration = num_frames / frame_rate  # Duration in seconds
    return duration

length = get_wav_length(file_path)
print(f"Length of the audio file: {length:.2f} seconds")