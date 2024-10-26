# based on https://huggingface.co/blog/fine-tune-whisper
# usage example: python3 training.py "small"

from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from datasets import DatasetDict, Dataset, load_dataset, load_from_disk
from compute_metrics import compute_metrics
import torch
import time
import argparse
import os

# parse command-line arguments
parser = argparse.ArgumentParser(description="Train Whisper model with a specified size.")
parser.add_argument("model_size", type=str, choices=["tiny", "small", "medium","large-v3"], help="Size of the Whisper model to use.")
args = parser.parse_args()

model_map = {
    "tiny": "openai/whisper-tiny", 
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
}
model_id = model_map[args.model_size]

# check if GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# load the processed dataset
dataset_path = f"../../data_processed/dataset_dict_{args.model_size}"
dataset_dict = load_from_disk(dataset_path)

train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["eval"]
test_dataset = dataset_dict["test"]

print("Data set loaded.")

model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.to(device)

# Disable use_cache for compatibility with gradient checkpointing
model.config.use_cache = False
model.generation_config.language = "English"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None


print(f"Model {model_id} set up finished.")

processor = WhisperProcessor.from_pretrained(model_id, language="English", task="transcribe")

# initialize the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
print("Data collator finished.")

# define training arguments based on the model size
def get_training_args(model_size):
    if model_size == "small":
        return Seq2SeqTrainingArguments(
            output_dir=f"../../trained_models/whisper-{model_size}-vanilla",
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            learning_rate=1.25e-5,
            warmup_steps=500,
            max_steps=8000,
            gradient_checkpointing=True,
            fp16=True,
            eval_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            save_total_limit=5,
        )
    elif model_size == "medium":
        return Seq2SeqTrainingArguments(
            output_dir=f"../../trained_models/whisper-{model_size}-vanilla",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=6.25e-6,
            warmup_steps=500,
            max_steps=14000,  # more steps for larger models
            gradient_checkpointing=True,
            fp16=True,
            eval_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            save_total_limit=5,
        )
    elif model_size == "large-v3":
        return Seq2SeqTrainingArguments(
            output_dir=f"../../trained_models/whisper-{model_size}-vanilla",
            per_device_train_batch_size=1,  # smaller batch size for larger models
            gradient_accumulation_steps=2,
            learning_rate=5e-6,
            warmup_steps=1000,
            max_steps=26000,  # more steps for large models
            gradient_checkpointing=True,
            fp16=True,
            eval_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=150,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            save_total_limit=5,
        )

training_args = get_training_args(args.model_size)

# check if a checkpoint exists in the output directory
def get_latest_checkpoint(output_dir):
    if os.path.isdir(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            # Sort checkpoints based on the integer part after 'checkpoint-' and return the latest one
            checkpoints.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
            return os.path.join(output_dir, checkpoints[0])
    return None

checkpoint = get_latest_checkpoint(training_args.output_dir)
if checkpoint:
    print(f"Resuming from checkpoint: {checkpoint}")
else:
    print("No checkpoint found. Starting fresh training.")

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=lambda p: compute_metrics(p, processor.feature_extractor),
    processing_class=processor.feature_extractor,  # Updated to use processing_class
)


processor.save_pretrained(training_args.output_dir)

print(f"Starting training for {model_id}...")
start_time = time.time()
trainer.train(resume_from_checkpoint=checkpoint)
end_time = time.time()
training_duration = end_time - start_time
print(f"Training completed in {training_duration // 3600} hours, "
      f"{(training_duration % 3600) // 60} minutes, and {training_duration % 60:.2f} seconds.")

# after training
print("Evaluating on the test dataset...")
predictions = trainer.predict(test_dataset=test_dataset)
print(predictions.metrics)