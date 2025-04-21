#!/usr/bin/env python3
import os
import torch

# -----------------------
# Patch PeftModel.forward to drop `num_items_in_batch`
# -----------------------
from peft import PeftModel
_orig_forward = PeftModel.forward
def _patched_forward(self, *args, **kwargs):
    kwargs.pop("num_items_in_batch", None)
    return _orig_forward(self, *args, **kwargs)
PeftModel.forward = _patched_forward

# --------------------------
# 1) Environment & Data
# --------------------------
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from compute_metrics import compute_metrics

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load your Set 2 Test dataset
test_ds = load_from_disk("/home/lian/data_processed/eval_dataset_ft_set2_test_small")
print(f"Test dataset size: {len(test_ds)}")

# --------------------------
# 2) Load base model + adapter
# --------------------------
# ① Load the original whisper-small
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
base_model.config.use_cache = False

# ② Attach your trained r=16,α=24 adapter
adapter_path = "/home/lian/data_processed/models/lora_personalized_speaker001_small32r64a"
model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
model.print_trainable_parameters()  # trainable% should be 0

# Load the processor saved alongside that adapter
processor = WhisperProcessor.from_pretrained(adapter_path, language="en", task="transcribe")

# --------------------------
# 3) Trainer for prediction
# --------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="/home/lian/tmp/test_run",  # won't write heavy checkpoints
    do_train=False,
    do_eval=False,
    do_predict=True,
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    ),
    tokenizer=processor.tokenizer,
    compute_metrics=lambda p: compute_metrics(p, processor.tokenizer)
)

# --------------------------
# 4) Run Prediction
# --------------------------
prediction_output = trainer.predict(test_ds)
metrics = prediction_output.metrics

print("===== Test Results =====")
for k, v in metrics.items():
    print(f"{k}: {v}")

