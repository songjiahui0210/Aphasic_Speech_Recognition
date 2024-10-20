# based on https://huggingface.co/blog/fine-tune-whisper

from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from datasets import DatasetDict, Dataset, load_dataset, load_from_disk
from compute_metrics import compute_metrics
import torch
import time
import argparse

# parse command-line arguments
parser = argparse.ArgumentParser(description="Train Whisper model with a specified size.")
parser.add_argument("model_size", type=str, choices=["tiny", "small", "medium", "large"], help="Size of the Whisper model to use.")
args = parser.parse_args()

model_map = {
    "tiny": "openai/whisper-tiny",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
}
model_id = model_map[args.model_size]

# check if GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# load the processed dataset
dataset_path = "../../data_processed/final_processed_dataset_small"
dataset_dict = load_from_disk(dataset_path)

train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["validation"]
test_dataset = dataset_dict["test"]

print("Data set loaded.")

model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.to(device)

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

training_args = Seq2SeqTrainingArguments(
    output_dir=f"../../trained_models/whisper-{args.model_size}-vanilla",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
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
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

print(f"Starting training for {model_id}...")
start_time = time.time()
trainer.train()
end_time = time.time()
training_duration = end_time - start_time
print(f"Training completed in {training_duration // 3600} hours, "
      f"{(training_duration % 3600) // 60} minutes, and {training_duration % 60:.2f} seconds.")

# after training
print("Evaluating on the test dataset...")
predictions = trainer.predict(test_dataset=test_dataset)
print(predictions.metrics)