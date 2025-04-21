from datasets import load_from_disk

train_dataset = load_from_disk("/scratch/liu.lian1/data_processed/train_dataset_filtered")
eval_dataset = load_from_disk("/scratch/liu.lian1/data_processed/eval_dataset_filtered")

none_samples = train_dataset.filter(lambda x: x["input_features"] is None)
print("Sample with problem:", len(none_samples))


for i, sample in enumerate(none_samples):
    if i >= 5:
        break
    print(f"Problem Sample {i}:")
    print(sample)
