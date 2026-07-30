#!/bin/bash
#SBATCH --job-name=AISIT_5_member
#SBATCH --partition=standard
#SBATCH --account=nemo
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#
micromamba activate AISIT2
python example_5_member.py
