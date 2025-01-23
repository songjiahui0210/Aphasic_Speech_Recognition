from transformers import WhisperForConditionalGeneration, AutoTokenizer, AutoProcessor
from huggingface_hub import HfApi

path_to_model_directory = "../trained_models/whisper-large-freezing-5" #change this path

# Load model and tokenizer
model = WhisperForConditionalGeneration.from_pretrained(path_to_model_directory)
tokenizer = AutoTokenizer.from_pretrained(path_to_model_directory)
processor = AutoProcessor.from_pretrained(path_to_model_directory)

# Define repository name
repo_name = "LitingZhou/whisper-large-freezing-5" # change the model name, don't change LitingZhou/

# Push to Hugging Face
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
processor.push_to_hub(repo_name)