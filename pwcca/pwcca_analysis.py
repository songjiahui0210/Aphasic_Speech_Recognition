import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_from_disk
from pwcca import compute_pwcca
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import Dataset
import os


# Path to the specific Arrow file you want to load
arrow_file_path = "../../data_processed/processed_audio_dataset_large-v3/train/data-00000-of-00098.arrow"

checkpoint_path = "pwcca_checkpoint.npz" 

# Load the dataset from just the single Arrow file
single_file_dataset = Dataset.from_file(arrow_file_path)

# Load models (Original and Fine-Tuned)
base_model_name = "openai/whisper-large-v3"
finetuned_model_path = "../../trained_models/whisper-large-v3-vanilla/checkpoint-14000"
device = "cuda" if torch.cuda.is_available() else "cpu"

original_model = WhisperForConditionalGeneration.from_pretrained(base_model_name).to(device)
finetuned_model = WhisperForConditionalGeneration.from_pretrained(finetuned_model_path).to(device)

processor = WhisperProcessor.from_pretrained(base_model_name, language="English", task="transcribe")

# Extract hidden states from each layer for a given sample
def extract_layer_outputs(model, inputs):
    # Add decoder_input_ids (use the start token from model config)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to(device)

    # Extract hidden states
    with torch.no_grad():
        outputs = model(
            input_features=inputs["input_features"].to(device),
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True
        )
    return outputs.encoder_hidden_states

# Compute PWCCA similarity for individual layers using compute_pwcca
def compute_pwcca_similarity(original_reps, finetuned_reps):
    similarities = []
    for i in range(len(original_reps)):
        original_layer = original_reps[i].squeeze().cpu().numpy().astype(np.float64)
        finetuned_layer = finetuned_reps[i].squeeze().cpu().numpy().astype(np.float64)

        # Try handling SVD non-convergence
        try:
            pwcca_score, _, _ = compute_pwcca(original_layer.T, finetuned_layer.T)
        except np.linalg.LinAlgError:
            print(f"SVD did not converge for layer {i}, skipping.")
            continue

        similarities.append(pwcca_score)
    return similarities

# Compute PWCCA similarity across all samples in the dataset
def compute_pwcca_similarity_for_dataset(processor, original_model, finetuned_model, dataset, checkpoint_path="pwcca_results.npz"):
    all_similarities = []
    
    # If a checkpoint exists, load the saved similarities
    if os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path, allow_pickle=True)
        all_similarities = list(data['similarities'])
        start_index = len(all_similarities)  # Start where the previous session left off
        print(f"Resuming from example {start_index}")
    else:
        start_index = 0

    for idx, example in enumerate(tqdm(dataset, desc="Calculating PWCCA")):
        if idx < start_index:
            continue  # Skip already processed examples

        # Check dataset structure
        if 'audio' not in example or 'array' not in example['audio']:
            print("Dataset structure issue detected at index:", idx, example)
            continue    

    #for idx, example in enumerate(tqdm(dataset[start_index:], desc="Calculating PWCCA", initial=start_index, total=len(dataset))):
        # Process audio to generate input features
        audio = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features

        inputs = {"input_features": input_features.to(device)}

        # Extract hidden states for original and fine-tuned models
        original_reps = extract_layer_outputs(original_model, inputs)
        finetuned_reps = extract_layer_outputs(finetuned_model, inputs)

        # Compute PWCCA similarity for this example
        similarities = compute_pwcca_similarity(original_reps, finetuned_reps)
        all_similarities.append(similarities)

        # Periodically save progress
        if idx % 50 == 0:  # Save every 50 examples
            np.savez(checkpoint_path, similarities=np.array(all_similarities))

    # Compute the average similarity across all examples for each layer
    avg_similarities = np.mean(np.array(all_similarities), axis=0)
    return avg_similarities

# Plot the PWCCA similarities for all layers
def plot_pwcca_similarity(layers, layerwise_similarities):
    plt.plot(layers, layerwise_similarities, marker='o', label='PWCCA Similarity')
    plt.title('Layer-wise PWCCA Similarity: Original vs Fine-tuned')
    plt.xlabel('Layer')
    plt.ylabel('Similarity')
    plt.grid(True)
    plt.legend()
    plt.savefig('large_pwcca_similarity_plot.png')

# Run analysis
#layerwise_similarities = compute_pwcca_similarity_for_dataset(processor, original_model, finetuned_model, single_file_dataset)
#layers = list(range(1, len(layerwise_similarities) + 1))
#plot_pwcca_similarity(layers, layerwise_similarities)

# Run analysis
layerwise_similarities = compute_pwcca_similarity_for_dataset(
    processor, original_model, finetuned_model, single_file_dataset, checkpoint_path="pwcca_results.npz"
)
layers = list(range(1, len(layerwise_similarities) + 1))
plot_pwcca_similarity(layers, layerwise_similarities)