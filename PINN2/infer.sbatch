#!/bin/bash
#SBATCH --job-name=AISIT_Test_Run
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#
module load jaspy
python infer.py   --checkpoint ./output/pinn_tracer.pt   --model /work/scratch-pw5/thopri/cmems_combined.nc   --model_year 1994 --model_month 10 --tracer_col oxy18
