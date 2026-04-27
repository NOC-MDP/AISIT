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
python infer.py   --checkpoint ./output/pinn_tracer.pt   --model /work/scratch-pw5/thopri/cmems_mod_arc_phy_my_topaz4_P1M_thetao-so_180.00W-179.88E_50.00N-90.00N_0.00-4000.00m_1991-01-01-2026-01-01.nc   --model_year 2007 --model_month 10 --tracer_col oxy18
