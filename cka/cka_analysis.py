import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt
from cka import feature_space_linear_cka

# Define CKA functions (as provided)
# Add your CKA functions (gram_linear, gram_rbf, center_gram, cka, etc.) here

# Load models
base_model_name = "openai/whisper-large-v3"
finetuned_model_path = "trained_models/whisper-large-v3-vanilla/checkpoint-14001"
device = "cuda" if torch.cuda.is_available() else "cpu"

original_model = WhisperForConditionalGeneration.from_pretrained(base_model_name).to(device)
finetuned_model = WhisperForConditionalGeneration.from_pretrained(finetuned_model_path).to(device)

processor = WhisperProcessor.from_pretrained(base_model_name, language="English", task="transcribe")

# Load preprocessed dataset
dataset_path = "data_processed/cka_test_dataset_dict_large"
dataset_dict = load_from_disk(dataset_path)
dataset = dataset_dict["train"]

# Extract hidden states from each encoder layer
def extract_layer_outputs(model, inputs):
    # Add decoder_input_ids with start token
    decoder_start_token_id = model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token_id]]).to(inputs["input_features"].device)
    with torch.no_grad():
        outputs = model(
            input_features=inputs["input_features"].to(device),
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True
        )
    # Return encoder hidden states for CKA
    if 'encoder_hidden_states' in outputs:
        return outputs.encoder_hidden_states
    else:
        raise ValueError("Encoder hidden states are not available. Check the model configuration.")

# Compute CKA similarity for each layer with detailed progress print statements
def compute_cka_for_layers(original_model, finetuned_model, dataset, processor, device):
    cka_similarities = []
    num_layers = original_model.config.encoder_layers

    for i in range(num_layers):
        layer_similarities = []  # Store CKA similarities for each example at this layer
        print(f"\nStarting processing for layer {i + 1}/{num_layers}...")

        for example_idx, example in enumerate(dataset):
            # Prepare input features
            inputs = {
                "input_features": torch.tensor(example["input_features"]).unsqueeze(0).to(device)
            }

            # Extract layer outputs for both models
            original_hidden_states = extract_layer_outputs(original_model, inputs)
            finetuned_hidden_states = extract_layer_outputs(finetuned_model, inputs)

            # Convert layer outputs to numpy arrays for CKA calculation
            original_layer_output = original_hidden_states[i].squeeze(0).cpu().numpy()
            finetuned_layer_output = finetuned_hidden_states[i].squeeze(0).cpu().numpy()

            # Compute linear CKA similarity
            cka_similarity = feature_space_linear_cka(original_layer_output, finetuned_layer_output)
            layer_similarities.append(cka_similarity)

            # Print progress for each input processed
            print(f"Layer {i + 1}/{num_layers}, Input {example_idx + 1}/{len(dataset)}: CKA similarity computed.")

        # Average CKA similarity across samples for the current layer
        avg_similarity = np.mean(layer_similarities)
        cka_similarities.append(avg_similarity)
        print(f"Completed layer {i + 1}/{num_layers}: Average CKA similarity = {avg_similarity:.4f}")

    print("\nAll layers processed.")
    # Print the final CKA similarities for all layers
    print("\nFinal CKA Similarities for All Layers:")
    for layer_index, similarity in enumerate(cka_similarities, start=1):
        print(f"Layer {layer_index}: CKA Similarity = {similarity:.4f}")
    return cka_similarities


# Plot CKA similarities for each layer
def plot_cka_similarities(cka_similarities):
    layers = list(range(1, len(cka_similarities) + 1))
    plt.plot(layers, cka_similarities, marker='o', label='CKA Similarity')
    plt.title('Layer-wise CKA Similarity: Original vs Fine-tuned')
    plt.xlabel('Layer')
    plt.ylabel('CKA Similarity')
    plt.grid(True)
    plt.legend()
    plt.savefig('cka_similarity_plot.png')
    plt.show()

# Generate heatmap for CKA similarities
def plot_cka_heatmap(cka_similarities):
    # Create a square matrix for the heatmap
    cka_matrix = np.array(cka_similarities).reshape((len(cka_similarities), 1))  # Assuming each layer comparison is separate

    plt.figure(figsize=(8, 6))
    sns.heatmap(cka_matrix, annot=True, cmap="viridis", cbar_kws={'label': 'CKA Similarity'})
    plt.xlabel("Fine-tuned Model Layer")
    plt.ylabel("Original Model Layer")
    plt.title("CKA Similarity Between Layers of Original and Fine-tuned Whisper Model")
    plt.savefig("cka_similarity_heatmap.png")
    plt.show()
# Run analysis
cka_similarities = compute_cka_for_layers(original_model, finetuned_model, dataset, processor, device)
plot_cka_similarities(cka_similarities)
plot_cka_heatmap(cka_similarities)
