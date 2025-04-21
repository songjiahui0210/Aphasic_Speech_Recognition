import os
import pandas as pd
from jiwer import wer, cer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip

# —— Configuration (absolute paths on Polaris) —— #
INPUT_CSV  = "/home/lian/data_processed/baseline_set1_whisper_small/baseline_small_results.csv"
OUTPUT_CSV = "/home/lian/data_processed/baseline_set1_whisper_small/baseline_small_results_with_scores.csv"

# —— Check input file —— #
if not os.path.isfile(INPUT_CSV):
    raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

# —— Read CSV without header, assign columns —— #
# The file has no header and even repeats a header line in the middle,
# so we read it all as data, name the columns, then drop any rows where
# “reference” or “prediction” equals the literal string "reference"/"prediction".
df = pd.read_csv(
    INPUT_CSV,
    header=None,
    names=["folder", "filename", "reference", "prediction", "orig_wer"],
    dtype=str
)

# —— Drop accidental header rows and any rows missing text —— #
df = df[(df["reference"] != "reference") & (df["prediction"] != "prediction")]
df = df.dropna(subset=["reference", "prediction"])

# —— Text‐cleaning pipeline —— #
cleaner = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip()
])

df["ref_clean"]  = df["reference"].apply(cleaner)
df["pred_clean"] = df["prediction"].apply(cleaner)

# —— Keep only non‐empty lines —— #
df_valid = df[
    (df["ref_clean"].str.strip() != "") &
    (df["pred_clean"].str.strip() != "")
].copy()

# —— Compute overall WER & CER —— #
overall_wer = wer(df_valid["ref_clean"].tolist(), df_valid["pred_clean"].tolist())

print(f"Total rows (incl. dropped): {len(df)}, Valid rows: {len(df_valid)}")
print(f"Overall WER: {overall_wer*100:.2f}%")

# —— Compute per‐row WER & CER —— #
df_valid["wer"] = df_valid.apply(
    lambda r: wer([r["ref_clean"]], [r["pred_clean"]]), axis=1
)

# —— Save results —— #
df_valid.to_csv(OUTPUT_CSV, index=False)
print(f"Per‐row WER/CER saved to: {OUTPUT_CSV}")

