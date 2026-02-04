#!/bin/bash
# Use conda environment with MLX installed
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate arxiv_daily
python main.py > "./logs/$(date +%F_%H-%M-%S).txt" 2>&1 
