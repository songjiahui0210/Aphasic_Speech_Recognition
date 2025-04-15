import os
import torch
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from compute_metrics import compute_metrics
from peft import LoraConfig, get_peft_model

# --------------------------------
# 1) Environment setup
# --------------------------------
torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------
# 2) Load preprocessed and filtered datasets
# --------------------------------
train_dataset = load_from_disk("../../data_processed/train_dataset_ft_set2_validation_small")
eval_dataset = load_from_disk("../../data_processed/eval_dataset_ft_set2_validation_small")

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size:  {len(eval_dataset)}")

# (Optional) For testing/debugging with smaller dataset
# subset_size = 500
# train_dataset = train_dataset.select(range(min(len(train_dataset), subset_size)))
# eval_dataset = eval_dataset.select(range(min(len(eval_dataset), subset_size)))

# --------------------------------
# 3) Load base model & configure LoRA
#    Note: If your processed data used 'whisper-small' for preprocessing,
#          you should also use 'openai/whisper-small' here to ensure tokenizer compatibility.
# --------------------------------
# 改这里
model_id = "../../models/whisper_lora_small"

whisper_model = WhisperForConditionalGeneration.from_pretrained(model_id)
whisper_model.config.use_cache = False  # Can reduce errors in some cases, but uses more VRAM

lora_config = LoraConfig(
    r=8,                   
    lora_alpha=16,         
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)
whisper_model = get_peft_model(whisper_model, lora_config)

whisper_model.print_trainable_parameters()

# If you only want to train LoRA, don't use the loop below;
# LoRA plugin automatically makes LoRA parameters trainable while freezing the base model.
# If you want full fine-tuning + LoRA, keep this loop.
# for param in whisper_model.parameters():
#     param.requires_grad = True

processor = WhisperProcessor.from_pretrained(model_id, language="en", task="transcribe")

# --------------------------------
# 4) Training hyperparameters
# --------------------------------
training_args = Seq2SeqTrainingArguments(
    # 改这
    output_dir="../../models/lora_validation_personalized_speaker001",  
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    warmup_steps=1000,
    max_steps=3000,
    # gradient_checkpointing=True,  # Optional: Use gradient checkpointing to save VRAM
    # Note: bf16=True only if GPU supports BF16, otherwise use fp16=True, bf16=False
    fp16=True,
    bf16=False,
    remove_unused_columns=False,
    evaluation_strategy="steps", 
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=3,
    predict_with_generate=True,
    generation_max_length=225  # Set maximum generation length here
)

# --------------------------------
# 5) Resume from checkpoint if available
# --------------------------------
checkpoint_dir = training_args.output_dir
latest_checkpoint = None

if os.path.isdir(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = os.path.join(
            checkpoint_dir,
            max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        )
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found. Starting from scratch.")
else:
    print("No checkpoint directory found. Training from scratch.")

# --------------------------------
# 6) DataCollator & Trainer
# --------------------------------
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=whisper_model.config.decoder_start_token_id
)

trainer = Seq2SeqTrainer(
    model=whisper_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    # Important: must use processor.tokenizer, not processor.feature_extractor
    tokenizer=processor.tokenizer,
    # If compute_metrics(p, tokenizer) internally uses tokenizer.decode,
    # then pass processor.tokenizer to it
    compute_metrics=lambda p: compute_metrics(p, processor.tokenizer)
)

# ========== Debug section ==========
# Check first batch integrity
try:
    train_dataloader = trainer.get_train_dataloader()
    first_batch = next(iter(train_dataloader))
    
    print("First batch keys:", first_batch.keys())
    if "input_features" in first_batch:
        print("Input features shape:", first_batch["input_features"].shape)
        print("Input features data type:", first_batch["input_features"].dtype)
        print("Input features contains NaN:", torch.isnan(first_batch["input_features"]).any())
    if "attention_mask" in first_batch:
        print("Attention mask shape:", first_batch["attention_mask"].shape)
    if "labels" in first_batch:
        print("Labels shape:", first_batch["labels"].shape)
    
    print("Training batch check successful!")
except Exception as e:
    print(f"Error checking training batch: {e}")

# --------------------------------
# 7) Start training
# --------------------------------
trainer.train(resume_from_checkpoint=latest_checkpoint)

# --------------------------------
# 8) Save model
# --------------------------------
trainer.save_model(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
print(f"LoRA fine-tuning saved to '{training_args.output_dir}'")