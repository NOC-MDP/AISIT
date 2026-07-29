#!/bin/bash
#SBATCH --job-name=AISIT2_Test_Run
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#
module load jaspy
python main.py --obs AISIT_Database_1.0.0_clima.parquet \
            --tracer_col Delta_O18_[permillle] \
            --date_col ISO8601_Datetime \
            --output_dir output_v2.0.0 \
            --epochs 100
