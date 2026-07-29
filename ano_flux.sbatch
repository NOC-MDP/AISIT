#!/bin/bash
#SBATCH --job-name=AISIT_Ano_Flux
#SBATCH --partition=standard
#SBATCH --account=nemo
#SBATCH --qos=standard
#SBATCH --time=03:00:00
#SBATCH --mem=128GB
#
micromamba activate AISIT2
python arctic_flux_ano.py
