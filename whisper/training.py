# based on https://huggingface.co/blog/fine-tune-whisper

from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from datasets import DatasetDict
from compute_metrics import compute_metrics


# load the processed dataset
dataset_path = "../../data_processed/final_processed_dataset_small"
dataset = Dataset.load_from_disk(dataset_path)

# split based on the 'split' column for train, validation, and test sets
train_dataset = dataset.filter(lambda example: example["split"] == "train")
eval_dataset = dataset.filter(lambda example: example["split"] == "validation")
test_dataset = dataset.filter(lambda example: example["split"] == "test")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.generation_config.language = "English"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None

# initialize the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

training_args = Seq2SeqTrainingArguments(
    output_dir="../../trained_models/whisper-small-vanilla",
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

processor = WhisperProcessor.from_pretrained(model_name, language="English", task="transcribe")

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

trainer.train()

# after training
predictions = trainer.predict(test_dataset=test_dataset)
print(predictions.metrics)