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
python main.py --obs ../data/bas_data/AISIT_Database_1.0.0_uv.parquet --tracer_col Delta_O18_[permillle] --date_col ISO8601_Datetime --model /work/scratch-pw5/thopri/cmems_combined.nc --model_year 2023 --model_month 2 --output_dir output_v1.0.0