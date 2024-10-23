source activate /home/zhou.lit/miniconda3/envs/trial

python3 -m debugpy --listen 5678 --wait-for-client ../generate_audio_chunks.py
