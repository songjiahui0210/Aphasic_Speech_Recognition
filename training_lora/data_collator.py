# based on https://huggingface.co/blog/fine-tune-whisper

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        
        for i, feature in enumerate(features):
            if feature is None:
                print(f"[Warning] Feature {i} is None.")
            elif "input_features" not in feature or feature["input_features"] is None:
                print(f"[Warning] Feature {i} missing input_features or is None.")

        
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
            if feature is not None and feature.get("input_features") is not None
        ]

        if not input_features:
            raise ValueError("All input_features are None. Check dataset or feature extractor pipeline.")

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        batch["attention_mask"] = torch.ones(batch["input_features"].shape[:-1], dtype=torch.long)

        label_features = [
            {"input_ids": feature["labels"]}
            for feature in features
            if feature is not None and feature.get("labels") is not None
        ]

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch