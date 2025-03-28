import evaluate
import numpy as np



def compute_metrics(pred, tokenizer):
    """
    Compute the Word Error Rate (WER) for model predictions.
    
    """
    # Load the WER metric from the `evaluate` library

    
    metric = evaluate.load("wer")

    # Extract predictions and label IDs from the output
    pred_ids = pred.predictions
    if isinstance(pred_ids, np.ndarray) and pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)  
    elif isinstance(pred_ids, np.ndarray) and pred_ids.ndim == 1:
        pred_ids = np.expand_dims(pred_ids, axis=0) 
    label_ids = pred.label_ids

    # debug
    print(f"[DEBUG] pred_ids type: {type(pred_ids)}, shape: {getattr(pred_ids, 'shape', None)}")
    if isinstance(pred_ids, np.ndarray):
        print("[DEBUG] pred_ids sample:", pred_ids[:20])  

    # debug
    for i, row in enumerate(pred_ids):
        if row.size == 0:  
            print(f"[DEBUG] Found empty pred_ids at index {i}!")

    # Replace -100 in labels ids with tokenizer's pad token ID. -100 is used to ignore indices in loss computations.
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_ids = label_ids.tolist()


    # Decode predictions and labels from ids to strings
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    print("[DEBUG] pred_str example:", pred_str[:2])
    print("[DEBUG] label_str example:", label_str[:2])
    # Calculate WER using the `evaluate` library
    wer_score = metric.compute(predictions=pred_str, references=label_str)

    # Return a dictionary with the WER result
    return {"wer": wer_score}