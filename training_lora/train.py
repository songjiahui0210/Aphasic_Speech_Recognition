from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
import torch

def load_datasets():
    return load_from_disk("../../data_processed/processed_dataset_large")

def train_model(model, dataset_dict):
    training_args = Seq2SeqTrainingArguments(
        output_dir="../../trained_models/whisper_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        eval_steps=500,
        save_steps=500,
        logging_steps=10,
        num_train_epochs=3,
        fp16=True,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["eval"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=None, decoder_start_token_id=model.config.decoder_start_token_id)
    )

    trainer.train()

if __name__ == "__main__":
    model = torch.load("../../models/whisper_lora.pth")
    dataset_dict = load_datasets()
    train_model(model, dataset_dict)