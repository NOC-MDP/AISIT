#!/bin/bash
#SBATCH --job-name=AISIT_Inference_Run
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#
micromamba activate AISIT2

python ODS_infer.py
