from datasets import load_from_disk
import os


# 1:small
dataset = load_from_disk("../../data_processed/processed_dataset_small_1")
print("Available columns in dataset:", dataset.column_names)


dataset_path = "../../data_processed/processed_dataset_small_new"
dataset = load_from_disk(dataset_path)

train_dataset = train_dataset.filter(lambda x: (x.get("input_features") is not None) and (len(x["input_features"]) > 0))
eval_dataset = eval_dataset.filter(lambda x: (x.get("input_features") is not None) and (len(x["input_features"]) > 0))


train_dataset.save_to_disk("../../data_processed/train_dataset_filtered_small")
eval_dataset.save_to_disk("../../data_processed/eval_dataset_filtered_small")