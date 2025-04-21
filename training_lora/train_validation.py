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
from peft import PeftModel
_orig_forward = PeftModel.forward
def _patched_forward(self, *args, **kwargs):
    kwargs.pop("num_items_in_batch", None)
    return _orig_forward(self, *args, **kwargs)
PeftModel.forward = _patched_forward
# --------------------------
# 1) 环境 & 数据
# --------------------------
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Validation 数据集路径
eval_ds = load_from_disk("/home/lian/data_processed/eval_dataset_ft_set2_validation_small")
print(f"Eval dataset size: {len(eval_ds)}")

# --------------------------
# 2) 加载模型 + adapter
# --------------------------
# 加载基础 Whisper-small
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
base_model.config.use_cache = False

# 将上一步训练好的 adapter 贴上来
adapter_path = "/home/lian/data_processed/models/lora_personalized_speaker001_small32r64a"
model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
model.print_trainable_parameters()  # 这里 trainable% 应该是 0

# Processor 必须和第一次训练 adapter 时用的一致
processor = WhisperProcessor.from_pretrained(adapter_path, language="en", task="transcribe")

# --------------------------
# 3) Trainer 只 eval，不 train
# --------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="/home/lian/tmp/val_run",  # 不会真的往里存 checkpoint
    do_train=False,
    do_eval=True,
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=eval_ds,
    data_collator=DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    ),
    tokenizer=processor.tokenizer,
    compute_metrics=lambda p: compute_metrics(p, processor.tokenizer)
)

# --------------------------
# 4) 运行 Evaluation
# --------------------------
metrics = trainer.evaluate()
print("===== Validation Results =====")
for k,v in metrics.items():
    print(f"{k}: {v}")

