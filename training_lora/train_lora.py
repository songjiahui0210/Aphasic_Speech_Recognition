import os
import torch
import numpy as np
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

# dataset_path = "../../data_processed/processed_dataset_large"
# dataset = load_from_disk(dataset_path)

# train_dataset = dataset.filter(lambda x: x["split"] == "train", num_proc=4)
# eval_dataset = dataset.filter(lambda x: x["split"] == "test", num_proc=4)

#subset size = 50
torch.cuda.empty_cache()

# check if GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

train_dataset = load_from_disk("../../data_processed/train_dataset_filtered")
eval_dataset = load_from_disk("../../data_processed/eval_dataset_filtered")

print(len(train_dataset))
#######
# subset_size = 3
# train_subset = train_dataset.select(range(min(len(train_dataset), 500)))
# eval_subset = eval_dataset.select(range(min(len(eval_dataset), 500)))

# train_dataset = train_subset
# eval_dataset = eval_subset

#####

model_id = "openai/whisper-large"  
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_id)
whisper_model.config.use_cache = False

lora_config = LoraConfig(
    r=24,                   
    lora_alpha=36,         
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)
whisper_model = get_peft_model(whisper_model, lora_config)

for param in whisper_model.parameters():
    param.requires_grad = True

processor = WhisperProcessor.from_pretrained(model_id, language="English", task="transcribe")

#reduce max steps for smaller sample size
# num_train_samples = len(train_dataset)
# num_epochs = 3  
# steps_per_epoch = num_train_samples // training_args.per_device_train_batch_size

#########

training_args = Seq2SeqTrainingArguments(
    output_dir="../../models/whisper_lora",  
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    warmup_steps=1000,
    max_steps=3000,
    # max_steps=300,
    gradient_checkpointing=True,
    fp16=False,
    bf16=True,
    evaluation_strategy="steps", 
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    # eval_steps = 50,
    # save_steps=50,
    # logging_steps=10,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=3,
    # save_total_limit=3,
    predict_with_generate=True
)

checkpoint_dir = "../../models/whisper_lora"
latest_checkpoint = None

if os.path.isdir(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = os.path.join(checkpoint_dir, max(checkpoints, key=lambda x: int(x.split("-")[-1])))
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found. Starting from scratch.")
else:
    print("No checkpoint directory found. Training from scratch.")


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor = processor,
    decoder_start_token_id = whisper_model.config.decoder_start_token_id
)

trainer = Seq2SeqTrainer(
    model=whisper_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,  
    compute_metrics=lambda p: compute_metrics(p, processor.tokenizer)
)


trainer.train(resume_from_checkpoint=latest_checkpoint)

whisper_model.save_pretrained("../../models/whisper_lora")
processor.save_pretrained("../../models/whisper_lora")

print("LoRA fine-tuning saved to '../../models/whisper_lora'")