#!/bin/bash

while true; do
    echo "Checking if each job is running..."

    # Check and submit each job individually if it is not running
    if ! squeue -u huang.luyi | grep -q "pwcca"; then
        echo "Job vanilla is not running. Resubmitting..."
        #sbatch select_gpu-vanilla.sh
        sbatch submit_job.sh
    else
        echo "Job vanilla is still running."
    fi

    echo "Waiting 20 minutes before checking again..."
    sleep 20m
done
