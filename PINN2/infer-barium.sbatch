#!/bin/bash
#SBATCH --job-name=AISIT_Inference_Run_Barium
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#
module load jaspy

# Define year range
START_YEAR=1992
END_YEAR=2017

OUTPUT_DIR=/gws/ssde/j25a/nemo/vol4/thopri/PINN_BARIUM/infer_output_ECCO

# Loop over all years and months
for YEAR in $(seq $START_YEAR $END_YEAR); do
    for MONTH in $(seq -w 1 12); do
        echo "Running inference for Year: $YEAR, Month: $MONTH"
        python infer.py \
            --checkpoint ./output_barium_v1.0.0/pinn_tracer.pt \
            --model /work/scratch-pw5/thopri/ECCO \
            --model_year $YEAR \
            --model_month $MONTH \
            --tracer_col Barium \
            --output_dir $OUTPUT_DIR

        # Check if the python command succeeded
        if [ $? -ne 0 ]; then
            echo "ERROR: Inference failed for Year: $YEAR, Month: $MONTH — skipping."
        fi
    done
done
