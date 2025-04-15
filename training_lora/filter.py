from datasets import load_from_disk
import os


# 1:M
dataset = load_from_disk("../../data_processed/processed_set2_validation_large")
print("Available columns in dataset:", dataset.column_names)


dataset_path = "../../data_processed/processed_set2_validation_large"
dataset = load_from_disk(dataset_path)

train_dataset = dataset.filter(lambda x: x["split"] == "train", num_proc=4)
eval_dataset = dataset.filter(lambda x: x["split"] == "test", num_proc=4)
# train_dataset = train_dataset.filter(lambda x: (x.get("input_features") is not None) and (len(x["input_features"]) > 0))
# eval_dataset = eval_dataset.filter(lambda x: (x.get("input_features") is not None) and (len(x["input_features"]) > 0))


train_dataset.save_to_disk("../../data_processed/train_dataset_ft_set2_validation_large")
eval_dataset.save_to_disk("../../data_processed/eval_dataset_ft_set2_validation_large")