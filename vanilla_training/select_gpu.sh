#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --job-name=gpu_run
#SBATCH --mem=8GB                     # increase memory if needed for your model
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

# Your code to run, for exampl
python3 training.py "large-v3"              
