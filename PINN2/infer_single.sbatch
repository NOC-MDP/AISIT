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

# Define year range
YEAR=2007
MONTH=10

OUTPUT_DIR=/home/users/thopri/AISIT/PINN2/infer_output

#
echo "Running inference for Year: $YEAR, Month: $MONTH"
python infer.py \
            --checkpoint ./output_lon_sin/pinn_tracer.pt \
            --model /work/scratch-pw5/thopri/cmems_combined.nc \
            --model_year $YEAR \
            --model_month $MONTH \
            --tracer_col oxy18 \
            --output_dir $OUTPUT_DIR

