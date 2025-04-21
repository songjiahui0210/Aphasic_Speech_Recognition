import torch
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model
from data_collator import DataCollatorSpeechSeq2SeqWithPadding

def debug_dataset(dataset):
    """Comprehensive dataset debugging function"""
    print("\n--- Dataset Debugging ---")
    print(f"Total samples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    
    # Check a few samples
    for i in range(min(3, len(dataset))):
        try:
            sample = dataset[i]
            print(f"\nSample {i}:")
            for key, value in sample.items():
                print(f"{key}: {type(value)}")
                if isinstance(value, (list, torch.Tensor, torch.Tensor)):
                    print(f"Shape/Length: {len(value) if isinstance(value, list) else value.shape}")
        except Exception as e:
            print(f"Error processing sample {i}: {e}")

def main():
    # Load datasets
    train_dataset = load_from_disk("/scratch/liu.lian1/data_processed/train_dataset_filtered")
    eval_dataset = load_from_disk("/scratch/liu.lian1/data_processed/eval_dataset_filtered")
    
    # Debug datasets
    debug_dataset(train_dataset)
    debug_dataset(eval_dataset)
    
    # Model and processor setup
    model_id = "openai/whisper-small"
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    processor = WhisperProcessor.from_pretrained(model_id, language="en", task="transcribe")
    
    # LoRA Configuration
    lora_config = LoraConfig(
        r=24,                   
        lora_alpha=36,         
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )
    whisper_model = get_peft_model(whisper_model, lora_config)
    
    # Data Collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=whisper_model.config.decoder_start_token_id
    )
    
    print("\n--- Debugging Complete ---")

if __name__ == "__main__":
    main()