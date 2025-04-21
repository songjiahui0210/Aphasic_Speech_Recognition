from datasets import load_from_disk
import numpy as np

# load dataset
train_dataset = load_from_disk("../../data_processed/train_dataset_filtered_small")

# check first ten

for i in range(min(10, len(train_dataset))):
    sample = train_dataset[i]
    print(f"样本 {i}:")
    if "input_features" in sample:
        if sample["input_features"] is None:
            print("  input_features 是 None")
        elif isinstance(sample["input_features"], list):
            print(f"  input_features 是列表，长度为: {len(sample['input_features'])}")
            if len(sample["input_features"]) > 0:
                print(f"  第一个元素类型: {type(sample['input_features'][0])}")
                if hasattr(sample["input_features"][0], "shape"):
                    print(f"  第一个元素形状: {sample['input_features'][0].shape}")
            else:
                print("  input_features 是空列表")
