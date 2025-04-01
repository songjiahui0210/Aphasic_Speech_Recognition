import evaluate
import numpy as np
import torch

def compute_metrics(pred, tokenizer):
    """
    Compute the Word Error Rate (WER) for model predictions.
    
    """
    # Load the WER metric from the `evaluate` library
    metric = evaluate.load("wer")
    pred_ids = pred.predictions
    if isinstance(pred_ids, np.ndarray) and pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)  
    elif isinstance(pred_ids, np.ndarray) and pred_ids.ndim == 1:
        pred_ids = np.expand_dims(pred_ids, axis=0)
    # Get label IDs
    label_ids = pred.label_ids

    # Handle empty input cases
    if pred_ids.size == 0:
        return {"wer": float('inf')}  # Return infinite error rate

    # Replace -100 in labels ids with tokenizer's pad token ID. -100 is used to ignore indices in loss computations.
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_ids = label_ids.tolist()

    try:
        # Attempt to decode predictions and labels
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Calculate WER
        wer_score = metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer_score}
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        print(f"Prediction shape: {pred_ids.shape if hasattr(pred_ids, 'shape') else 'unknown'}")
        print(f"Label shape: {label_ids.shape if hasattr(label_ids, 'shape') else 'unknown'}")
        # Return placeholder value in case of error
        return {"wer": 999.0}