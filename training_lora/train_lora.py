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
    r=32,                   
    lora_alpha=32,         
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
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    warmup_steps=1000,
    max_steps=3000,
    # max_steps=300,
    gradient_checkpointing=True,
    fp16=False,
    bf16=True,
    evaluation_strategy="steps", 
    eval_steps=5000,
    save_steps=5000,
    logging_steps=500,
    # eval_steps = 50,
    # save_steps=50,
    # logging_steps=10,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=5,
    # save_total_limit=3,
    predict_with_generate=True
)

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

torch.cuda.empty_cache()

trainer.train()

whisper_model.save_pretrained("../../models/whisper_lora")
processor.save_pretrained("../../models/whisper_lora")

print("LoRA fine-tuning saved to '../../models/whisper_lora'")