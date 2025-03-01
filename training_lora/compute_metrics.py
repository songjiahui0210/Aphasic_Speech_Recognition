import evaluate



def compute_metrics(pred, tokenizer):
    """
    Compute the Word Error Rate (WER) for model predictions.
    
    """
    # Load the WER metric from the `evaluate` library
    metric = evaluate.load("wer")

    # Extract predictions and label IDs from the output
    pred_ids = pred.predictions.argmax(-1)
    label_ids = pred.label_ids

    # Replace -100 in labels ids with tokenizer's pad token ID. -100 is used to ignore indices in loss computations.
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels from ids to strings
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Calculate WER using the `evaluate` library
    wer_score = metric.compute(predictions=pred_str, references=label_str)

    # Return a dictionary with the WER result
    return {"wer": wer_score}