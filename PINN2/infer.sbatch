#!/bin/bash
#SBATCH --job-name=AISIT_Test_Run
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#
module load jaspy

# Define year range
START_YEAR=1991
END_YEAR=2025

OUTPUT_DIR=/gws/ssde/j25a/nemo/vol4/thopri/PINN/infer_output

# Loop over all years and months
for YEAR in $(seq $START_YEAR $END_YEAR); do
    for MONTH in $(seq -w 1 12); do
        echo "Running inference for Year: $YEAR, Month: $MONTH"
        python infer.py \
            --checkpoint ./output_lon_sin/pinn_tracer.pt \
            --model /work/scratch-pw5/thopri/cmems_combined.nc \
            --model_year $YEAR \
            --model_month $MONTH \
            --tracer_col oxy18 \
            --output_dir $OUTPUT_DIR
        
        # Check if the python command succeeded
        if [ $? -ne 0 ]; then
            echo "ERROR: Inference failed for Year: $YEAR, Month: $MONTH — skipping."
        fi
    done
done

echo "All inference runs complete. Combining NetCDF files with NCO..."

# Combine all output .nc files into one along the time dimension
ncrcat ${OUTPUT_DIR}/*.nc ${OUTPUT_DIR}/combined_1991_2025.nc

if [ $? -eq 0 ]; then
    echo "Successfully combined all files into: ${OUTPUT_DIR}/combined_1991_2025.nc"
else
    echo "ERROR: NCO combination failed. Check individual files in ${OUTPUT_DIR}/"
fi
