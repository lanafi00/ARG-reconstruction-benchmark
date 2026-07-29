#!/bin/bash
# Job Name
#SBATCH --job-name=my_simulation
#SBATCH --array=0-9
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --partition=tier1q

mkdir -p logs

source /apps/software/gcc-12.1.0/miniconda3/24.9.2/etc/profile.d/conda.sh
conda activate singer-snakemake-env
python official_sim_gen2.py $SLURM_ARRAY_TASK_ID
