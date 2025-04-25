#!/bin/bash

# Run demo for Aphasic Speech Recognition with Whisper LoRA
# This script provides a simple way to run the demo with different configurations

# Set script to exit immediately if any command fails
set -e

# Default values
MODEL_SIZE="small"
DATA_SUBSET=100
DEMO_STEPS=10
LORA_R=8
LORA_ALPHA=16
TEST_AUDIO="../../data_processed/audios/ACWT/ACWT01a_144.813_2.78.wav"

# Create necessary directories
mkdir -p demo_data/outputs

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}    Aphasic Speech Recognition Demo Runner      ${NC}"
echo -e "${BLUE}================================================${NC}"

# Check if the user provided a specific model size
if [ "$1" != "" ]; then
    MODEL_SIZE=$1
    echo -e "${YELLOW}Using specified model size: $MODEL_SIZE${NC}"
fi

# Check Python and dependencies
echo -e "\n${GREEN}Checking environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

echo "Python version:"
python3 --version

# Check if data preparation is needed
echo -e "\n${GREEN}Checking data...${NC}"
if [ ! -f "demo_data/demo_subset.csv" ]; then
    echo "Creating data subset for demo..."
    python3 create_data_subset.py --num_speakers 5 --samples_per_speaker 20 --output_dir demo_data
else
    echo "Using existing demo subset"
fi

# Check if required files exist
if [ ! -f "demo_run.py" ]; then
    echo "Error: demo_run.py not found"
    exit 1
fi

# Run the demo
echo -e "\n${GREEN}Starting demo with Whisper-${MODEL_SIZE}...${NC}"
echo -e "Will run for ${DEMO_STEPS} steps with ${DATA_SUBSET} data samples"
echo -e "LoRA configuration: r=${LORA_R}, alpha=${LORA_ALPHA}"
echo -e "${BLUE}================================================${NC}"

# Execute the demo script
python3 demo_run.py \
    --model_size $MODEL_SIZE \
    --data_subset $DATA_SUBSET \
    --demo_steps $DEMO_STEPS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --test_audio $TEST_AUDIO

# Check if the demo finished successfully
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Demo completed successfully!${NC}"
    echo -e "Check demo_data/demo_output for model checkpoints"
else
    echo -e "\n${YELLOW}Demo encountered an issue${NC}"
    echo "Check the error messages above for details"
fi

echo -e "\n${BLUE}================================================${NC}"
echo -e "${BLUE}             Demo Run Complete                  ${NC}"
echo -e "${BLUE}================================================${NC}"