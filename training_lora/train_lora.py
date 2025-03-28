import os
import torch
import numpy as np
from datasets import load_from_disk  
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from compute_metrics import compute_metrics
from peft import LoraConfig, get_peft_model
import types


# Clear GPU cache
torch.cuda.empty_cache()

# Check device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load dataset
train_dataset = load_from_disk("../../data_processed/train_dataset_filtered")
eval_dataset = load_from_disk("../../data_processed/eval_dataset_filtered")
print(len(train_dataset))

# Load model
model_id = "openai/whisper-small"  
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_id)
whisper_model.config.use_cache = False

# Apply LoRA
lora_config = LoraConfig(
    r=8,                   
    lora_alpha=18,         
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)
whisper_model = get_peft_model(whisper_model, lora_config)

for param in whisper_model.parameters():
    param.requires_grad = True

# Load processor
processor = WhisperProcessor.from_pretrained(model_id, language="en", task="transcribe")

# Training arguments
output_dir = "../../models/whisper_lora_small"  
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,  
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    warmup_steps=1000,
    max_steps=3000,
    gradient_checkpointing=True,
    fp16=False,
    bf16=True,
    evaluation_strategy="steps", 
    eval_steps=500,
    save_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=3,
    predict_with_generate=True
)

# Resume from checkpoint
def get_resume_checkpoint(output_dir):
 
    if not os.path.isdir(output_dir):
        return None

    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isfile(os.path.join(output_dir, d, "trainer_state.json"))
    ]
    if not checkpoints:
        return None

    latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return latest


resume_checkpoint = get_resume_checkpoint(output_dir)
if resume_checkpoint:
    print(f"Resuming training from checkpoint: {resume_checkpoint}")
else:
    print("No checkpoint found. Training from scratch.")

# Data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor = processor,
    decoder_start_token_id = whisper_model.config.decoder_start_token_id
)

# Trainer
trainer = Seq2SeqTrainer(
    model=whisper_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=processor,
    compute_metrics=lambda p: compute_metrics(p, processor.tokenizer)
)

def patched_load_rng_state(self, checkpoint_path):
    import os
    import torch
    rng_file = os.path.join(checkpoint_path, "rng_state.pth")
    if os.path.isfile(rng_file):
        try:
            rng_state = torch.load(rng_file, weights_only=False)
            self._rng_state = rng_state
            print(f"[DEBUG] RNG state loaded from: {rng_file}")
        except Exception as e:
            print(f"[WARNING] Failed to load rng_state.pth: {e}")

trainer._load_rng_state = types.MethodType(patched_load_rng_state, trainer)


# Custom callback for manual checkpoint
class SaveManualCheckpointCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 10:
            checkpoint_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-manual")
            os.makedirs(checkpoint_dir, exist_ok=True)
            trainer.save_model(checkpoint_dir)  
            processor.save_pretrained(checkpoint_dir)
            print(f"[DEBUG] Manual checkpoint saved at: {checkpoint_dir}")

trainer.add_callback(SaveManualCheckpointCallback())

trainer.train(resume_from_checkpoint=resume_checkpoint)
# Save final checkpoint (transformers format)
final_checkpoint_path = os.path.join(output_dir, f"checkpoint-{trainer.state.global_step}")
trainer.save_model(final_checkpoint_path)
print(f"Final checkpoint saved at {final_checkpoint_path}")

# # Manual save with torch
# manual_checkpoint_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-final")
# os.makedirs(manual_checkpoint_dir, exist_ok=True)
# torch.save(whisper_model.state_dict(), os.path.join(manual_checkpoint_dir, "pytorch_model.bin"))
# # torch.save(trainer.state, os.path.join(manual_checkpoint_dir, "trainer_state.pt"))
# print(f"[DEBUG] Manual final checkpoint saved to: {manual_checkpoint_dir}")

# Save model and processor for inference
whisper_model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"LoRA fine-tuning saved to '{output_dir}'")