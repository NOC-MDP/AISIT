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
python main.py --obs ../data/bas_data/pre_alpha_AISIT_0.1.4_new_headers_uv.csv --tracer_col oxy18 --date_col year --model /work/scratch-pw5/thopri/cmems_combined.nc --model_year 2023 --model_month 2 --output_dir output_lon_sin