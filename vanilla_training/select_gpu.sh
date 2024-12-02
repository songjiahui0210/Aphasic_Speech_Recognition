#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --job-name=gpu_run
#SBATCH --mem=8GB                     # increase memory if needed for your model
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

echo "Job started with job ID $SLURM_JOB_ID"
python3 training.py "large-v3"
echo "Job completed."
             
