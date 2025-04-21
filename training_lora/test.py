from datasets import load_from_disk
import numpy as np

# load dataset
train_dataset = load_from_disk("../../data_processed/train_dataset_filtered_small")

# check the first ten

for i in range(min(10, len(train_dataset))):
    sample = train_dataset[i]
    print(f"Sample {i}:")
    if "input_features" in sample:
        if sample["input_features"] is None:
            print("  input_features is None")
        elif isinstance(sample["input_features"], list):
            print(f"  input_features is length，length: {len(sample['input_features'])}")
            if len(sample["input_features"]) > 0:
                print(f"  The first element type: {type(sample['input_features'][0])}")
                if hasattr(sample["input_features"][0], "shape"):
                    print(f"  The first element shape: {sample['input_features'][0].shape}")
            else:
                print("  input_features is empty list")
