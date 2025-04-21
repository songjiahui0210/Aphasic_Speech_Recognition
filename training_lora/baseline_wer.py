import os
import torch
import pandas as pd
import librosa
import inflect
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation

def convert_numbers_to_words(text):
    """
    Convert digit numbers into words (e.g., 2002 -> two thousand and two).
    """
    p = inflect.engine()
    words = text.split()
    out = []
    for w in words:
        if w.isdigit():
            try:
                w = p.number_to_words(w)
            except inflect.NumOutOfRangeError:
                pass
        out.append(w)
    return " ".join(out)

def calculate_wer(csv_path, audio_root, model, processor, detailed_csv, device):
    """
    Calculate per-file WER and save detailed results to a CSV.
    """
    df = pd.read_csv(csv_path)
    first_write = True

    with open(detailed_csv, "w") as _:
        pass  # truncate existing file

    for _, row in df.iterrows():
        file_name     = row["file_cut"]
        transcription = row["transcriptions"]
        folder_name   = row["folder_name"]
        audio_path    = os.path.join(audio_root, folder_name, file_name)

        if not os.path.exists(audio_path):
            print(f"[WARN] Audio not found: {audio_path}")
            continue

        # load audio
        wav, sr = librosa.load(audio_path, sr=16000)
        inputs  = processor(wav, sampling_rate=sr, return_tensors="pt")
        feats   = inputs.input_features.to(device)

        # generate
        with torch.no_grad():
            ids = model.generate(feats)
        pred = processor.batch_decode(ids, skip_special_tokens=True)[0]

        # normalize
        norm = Compose([ToLowerCase(), RemovePunctuation()])
        ref_clean  = norm(transcription)
        pred_clean = norm(pred)
        pred_clean = convert_numbers_to_words(pred_clean)

        score = wer([ref_clean], [pred_clean])

        out = {
            "folder":     folder_name,
            "file":       file_name,
            "reference":  ref_clean,
            "prediction": pred_clean,
            "wer":        score
        }
        pd.DataFrame([out]).to_csv(
            detailed_csv, mode="a", header=first_write, index=False
        )
        first_write = False

    print(f"[INFO] Details written to {detailed_csv}")

def calculate_overall_wer_from_csv(detailed_csv):
    """
    Read the detailed CSV and compute overall WER.
    """
    df = pd.read_csv(detailed_csv)
    refs  = df["reference"].tolist()
    preds = df["prediction"].tolist()
    return wer(refs, preds)

def run_baseline_evaluation(model_size, csv_path, audio_root, results_folder):
    """
    Run baseline WER on a single CSV (used for both validation and test).
    """
    torch.cuda.empty_cache()
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model_id  = f"openai/whisper-{model_size}"
    print(f"[INFO] Loading model {model_id} on {device}...")
    model     = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    processor = WhisperProcessor.from_pretrained(model_id, language="en", task="transcribe")

    os.makedirs(results_folder, exist_ok=True)
    detailed_csv = os.path.join(results_folder, f"baseline_{model_size}_results.csv")

    print(f"[INFO] Processing {csv_path} ...")
    calculate_wer(csv_path, audio_root, model, processor, detailed_csv, device)

    overall = calculate_overall_wer_from_csv(detailed_csv)
    print(f"[RESULT] {model_size} WER on {os.path.basename(csv_path)}: {overall*100:.2f}%")

    # cleanup
    del model
    torch.cuda.empty_cache()
    return overall

if __name__ == "__main__":
     # Common settings
    validation_csv = "/home/lian/data_processed/set1_w_cohort.csv"
    test_csv       = validation_csv
    audio_root     = "/home/lian/data_processed/audios"

    for model_size in ["small", "medium"]:
        results_folder = f"/home/lian/data_processed/baseline_set1_whisper_{model_size}"
        print(f"\n===== Running baseline for whisper-{model_size} =====")
        val_wer = run_baseline_evaluation(
            model_size, validation_csv, audio_root, results_folder
        )
        test_wer = run_baseline_evaluation(
            model_size, test_csv, audio_root, results_folder
        )
        print(
            f">>> whisper-{model_size} → Validation WER: {val_wer*100:.2f}%  "
            f"Test WER: {test_wer*100:.2f}%"
    )

