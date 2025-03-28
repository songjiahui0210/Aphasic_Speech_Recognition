import os
import numpy as np
from datasets import load_from_disk
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data path configuration
input_dataset_path = "../../data_processed/processed_dataset_small_1"  # Original dataset path
output_train_path = "../../data_processed/train_dataset_filtered_small"  # Output training set path
output_eval_path = "../../data_processed/eval_dataset_filtered_small"   # Output validation set path

# --------------------------------
# 1) Load original dataset
# --------------------------------
logger.info(f"Loading dataset: {input_dataset_path}")
try:
    dataset = load_from_disk(input_dataset_path)
    logger.info(f"Dataset loaded successfully, total samples: {len(dataset)}")
    logger.info(f"Available columns: {dataset.column_names}")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    exit(1)

# --------------------------------
# 2) Split training and validation sets (if not already split)
# --------------------------------
if "train" in dataset and "eval" in dataset:
    logger.info("Using existing train/validation split")
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]
else:
    logger.info("Dataset not split, performing train/validation split")
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

logger.info(f"Training set size: {len(train_dataset)}")
logger.info(f"Validation set size: {len(eval_dataset)}")

# --------------------------------
# 3) Convert nested lists to numpy arrays
# --------------------------------
def convert_nested_lists_to_numpy(example):
    """Convert nested lists to numpy arrays"""
    if "input_features" in example and example["input_features"] is not None:
        if isinstance(example["input_features"], list):
            try:
                example["input_features"] = np.array(example["input_features"], dtype=np.float32)
                # Log shapes of some samples to check if conversion is correct
                if np.random.random() < 0.001:  # Randomly log ~0.1% of sample shapes
                    logger.info(f"Sample input_features shape example: {example['input_features'].shape}")
            except Exception as e:
                logger.warning(f"Error converting input_features: {e}")
    return example

logger.info("Converting nested lists to numpy arrays...")
train_dataset = train_dataset.map(convert_nested_lists_to_numpy)
eval_dataset = eval_dataset.map(convert_nested_lists_to_numpy)

# --------------------------------
# 4) Check and filter invalid samples
# --------------------------------
def validate_features(example):
    """Check if sample features are valid"""
    valid = True
    if "input_features" not in example or example["input_features"] is None:
        valid = False
    elif isinstance(example["input_features"], np.ndarray):
        if example["input_features"].size == 0:
            valid = False
    elif isinstance(example["input_features"], list) and len(example["input_features"]) == 0:
        valid = False
    
    return valid

# Count valid samples
logger.info("Counting valid samples...")
train_valid_count = sum(validate_features(sample) for sample in train_dataset)
eval_valid_count = sum(validate_features(sample) for sample in eval_dataset)
logger.info(f"Valid training samples: {train_valid_count}/{len(train_dataset)} ({train_valid_count/len(train_dataset)*100:.2f}%)")
logger.info(f"Valid validation samples: {eval_valid_count}/{len(eval_dataset)} ({eval_valid_count/len(eval_dataset)*100:.2f}%)")

# Filter invalid samples
if train_valid_count < len(train_dataset) or eval_valid_count < len(eval_dataset):
    logger.info("Invalid samples found, starting filtering...")
    
    # Size before filtering
    logger.info(f"Before filtering - Training set size: {len(train_dataset)}")
    logger.info(f"Before filtering - Validation set size: {len(eval_dataset)}")
    
    # Perform filtering
    train_dataset = train_dataset.filter(validate_features)
    eval_dataset = eval_dataset.filter(validate_features)
    
    # Size after filtering
    logger.info(f"After filtering - Training set size: {len(train_dataset)}")
    logger.info(f"After filtering - Validation set size: {len(eval_dataset)}")
else:
    logger.info("All samples are valid, no filtering needed")

# --------------------------------
# 5) Check other data quality issues (optional)
# --------------------------------
# Check if labels exist and are valid
if "labels" in train_dataset.column_names:
    # Check for empty labels
    empty_labels_train = sum(1 for x in train_dataset if x["labels"] is None or len(x["labels"]) == 0)
    empty_labels_eval = sum(1 for x in eval_dataset if x["labels"] is None or len(x["labels"]) == 0)
    
    logger.info(f"Empty labels samples in training set: {empty_labels_train}")
    logger.info(f"Empty labels samples in validation set: {empty_labels_eval}")
    
    # Filter empty labels samples if there are many
    if empty_labels_train > 0 or empty_labels_eval > 0:
        logger.info("Filtering empty labels samples...")
        train_dataset = train_dataset.filter(lambda x: x["labels"] is not None and len(x["labels"]) > 0)
        eval_dataset = eval_dataset.filter(lambda x: x["labels"] is not None and len(x["labels"]) > 0)
        logger.info(f"After filtering empty labels - Training set size: {len(train_dataset)}")
        logger.info(f"After filtering empty labels - Validation set size: {len(eval_dataset)}")

# --------------------------------
# 6) Save processed datasets
# --------------------------------
logger.info(f"Saving processed training set to: {output_train_path}")
train_dataset.save_to_disk(output_train_path)

logger.info(f"Saving processed validation set to: {output_eval_path}")
eval_dataset.save_to_disk(output_eval_path)

logger.info("Data preprocessing and filtering complete!")

# Output some statistics (optional)
logger.info("======= Final Dataset Statistics =======")
logger.info(f"Training set samples: {len(train_dataset)}")
logger.info(f"Validation set samples: {len(eval_dataset)}")
if "input_features" in train_dataset[0]:
    if isinstance(train_dataset[0]["input_features"], np.ndarray):
        logger.info(f"input_features shape example: {train_dataset[0]['input_features'].shape}")
    else:
        logger.info(f"input_features type: {type(train_dataset[0]['input_features'])}")
if "labels" in train_dataset[0]:
    logger.info(f"labels example: {train_dataset[0]['labels'][:10]}...")