#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --job-name=whisper_large_30
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err


# log GPU usage every 1 hour
while true; do
    nvidia-smi >> gpu_usage_large_30.log
    sleep 3600
done &

python3 training.py "large" --freeze_layers 30
# python3 training.py "small"

# re-submit the job if not complete
if [ $? -ne 0 ]; then
    sbatch $0
fi