#!/bin/bash
#SBATCH --job-name="MWC_ano"
#SBATCH --time=4:00:00
#SBATCH --mem=200Gb
#SBATCH --account=nemo
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH -o %j.out
#SBATCH -e %j.err

micromamba activate AISIT2

python MW_anomaly2.py
