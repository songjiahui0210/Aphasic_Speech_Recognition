nvidia-smi
module purge
module load discovery
module load python/3.8.1 
module load anaconda3/3.7 
module load ffmpeg/20190305 
source activate /work/van-speech-nlp/jindaznb/slamenv/
which python

source activate /work/van-speech-nlp/jindaznb/slamenv/

python3 -m debugpy --listen 5678 --wait-for-client ../transcribe.py
