from datasets import load_from_disk
import os


# 1:small
dataset = load_from_disk("../../data_processed/processed_dataset_small_new")
print("Available columns in dataset:", dataset.column_names)


dataset_path = "../../data_processed/processed_dataset_small_new"
dataset = load_from_disk(dataset_path)

train_dataset = dataset.filter(lambda x: x["split"] == "train", num_proc=4)
eval_dataset = dataset.filter(lambda x: x["split"] == "test", num_proc=4)

train_dataset.save_to_disk("../../data_processed/train_dataset_filtered_small")
eval_dataset.save_to_disk("../../data_processed/eval_dataset_filtered_small")