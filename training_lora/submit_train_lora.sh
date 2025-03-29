#!/bin/bash
#SBATCH --job-name=whisper_lora_train      
#SBATCH --nodes=1                         
#SBATCH --ntasks=1                         
#SBATCH --cpus-per-task=4                  
#SBATCH --mem=64G                          
#SBATCH --time=08:00:00                   
#SBATCH --partition=gpu                    
#SBATCH --gres=gpu:1                       
#SBATCH --output=logs/output_%j.log        
#SBATCH --error=logs/error_%j.log          
#SBATCH --mail-user=$song.jiahui@northeastern.edu 
#SBATCH --mail-type=END,FAIL               

module load cuda/12.1

source activate /home/song.jiahui/envs/pylangacq_env

cd /scratch/song.jiahui/Aphasic_Speech_Recognition/training_lora

python3 train_lora.py