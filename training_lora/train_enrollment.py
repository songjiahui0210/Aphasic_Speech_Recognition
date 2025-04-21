#!/usr/bin/env python3
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
from peft import LoraConfig, get_peft_model, PeftModel

# ==============================
# 修补 PeftModel.forward，丢弃 num_items_in_batch
# ==============================
_orig_forward = PeftModel.forward
def _patched_forward(self, *args, **kwargs):
    kwargs.pop("num_items_in_batch", None)
    return _orig_forward(self, *args, **kwargs)
PeftModel.forward = _patched_forward

# --------------------------------
# 1) 环境 & 设备
# --------------------------------
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------
# 2) 加载 Set2 Enrollment 数据
# --------------------------------
train_dataset = load_from_disk("/home/lian/data_processed/train_dataset_ft_set2_enrollment_small")
eval_dataset  = load_from_disk("/home/lian/data_processed/eval_dataset_ft_set2_enrollment_small")
print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

# --------------------------------
# 3) 加载基础模型 + 初始化 LoRA
# --------------------------------
# ① 先加载在 Set1 上训练好的 cohort adapter（r=8, α=16）
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
base_model.config.use_cache = False
adapter_cohort = "/home/lian/data_processed/models/lora_personalized_speaker001"
model = PeftModel.from_pretrained(base_model, adapter_cohort).to(device)

# ② 再给它贴上这次要训练的新 adapter（r=16, α=24）
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=24,
    lora_dropout=0.1,
    target_modules=["q_proj","v_proj"],
    bias="none"
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")

# --------------------------------
# 4) TrainingArguments
# --------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="/home/lian/data_processed/models/lora_personalized_speaker001_small32r64a",
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    warmup_steps=1000,
    max_steps=3000,

    fp16=True,
    remove_unused_columns=False,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=3,
    predict_with_generate=True,
    generation_max_length=225,
)

# --------------------------------
# 5) DataCollator & Trainer
# --------------------------------
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
    compute_metrics=lambda p: compute_metrics(p, processor.tokenizer)
)

# --------------------------------
# 6) 开始训练
# --------------------------------
trainer.train()

# --------------------------------
# 7) 保存 Adapter
# --------------------------------
trainer.save_model(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
print(f"✅ Personalized Enrollment Adapter saved to {training_args.output_dir}")

