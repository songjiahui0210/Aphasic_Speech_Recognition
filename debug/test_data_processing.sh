source activate /home/zhou.lit/miniconda3/envs/trial

python3 -m debugpy --listen 5678 --wait-for-client ../data_processing.py "../../data_processed/transcripts/ACWT"