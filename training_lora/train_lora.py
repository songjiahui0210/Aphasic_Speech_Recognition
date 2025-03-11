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

train_dataset = load_from_disk("../../data_processed/train_dataset_filtered")
eval_dataset = load_from_disk("../../data_processed/eval_dataset_filtered")



model_id = "openai/whisper-large"  
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_id)

lora_config = LoraConfig(
    r=32,                   
    lora_alpha=32,         
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)
whisper_model = get_peft_model(whisper_model, lora_config)

processor = WhisperProcessor.from_pretrained(model_id, language="English", task="transcribe")

training_args = Seq2SeqTrainingArguments(
    output_dir="../../models/whisper_lora",  
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    warmup_steps=1000,
    max_steps=14000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps", 
    eval_steps=1000,
    save_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=5,
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

trainer.train()

whisper_model.save_pretrained("../../models/whisper_lora")
processor.save_pretrained("../../models/whisper_lora")

print("LoRA fine-tuning saved to '../../models/whisper_lora'")