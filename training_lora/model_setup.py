import torch
from torch import nn
import numpy as np
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LoRALayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.low_rank_u = nn.Parameter(torch.rand(out_features, rank))
        self.low_rank_v = nn.Parameter(torch.rand(rank, in_features))
        nn.init.kaiming_uniform_(self.low_rank_u, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.low_rank_v, a=np.sqrt(5))

    def forward(self, x):
        low_rank = self.low_rank_u @ self.low_rank_v
        return self.linear(x) + low_rank @ x

def modify_whisper_with_lora(model, layer_index, rank):
    # Access the specific layer in the encoder
    # target_layer = whisper_model.model.encoder.layers[layer_index].self_attn.out_proj
    # in_features = target_layer.in_features
    # out_features = target_layer.out_features
    
    # # Create the LoRA layer
    # lora_layer = LoRALayer(in_features, out_features, rank)
    
    # # Replace the original layer with LoRA layer
    # whisper_model.model.encoder.layers[layer_index].self_attn.out_proj = lora_layer
    lora_config = LoraConfig(
        r=rank,  # LoRA rank
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,  # Dropout
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
        bias="none"
    )

    return get_peft_model(model, lora_config)


model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large')
model = modify_whisper_with_lora(model, layer_index=0, rank=32)

from datasets import load_dataset

# Load a dataset (custom dataset or Hugging Face's Common Voice)
dataset = load_dataset("your_dataset", split="train")  # Replace with actual dataset name

from transformers import Seq2SeqTrainingArguments

Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    warmup_steps=1000,
    max_steps=14000,  
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

from transformers import Seq2SeqTrainer

processor = WhisperProcessor.from_pretrained(model_id, language="English", task="transcribe")

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

train_dataset = ""
eval_dataset = ""

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=lambda p: compute_metrics(p, processor.tokenizer),
    tokenizer=processor.feature_extractor,
)


trainer.train()

model.save_pretrained("../../models/whisper_lora")