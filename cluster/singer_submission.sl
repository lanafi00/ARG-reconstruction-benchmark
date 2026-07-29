#!/bin/bash
# Job Name
#SBATCH --job-name=my_singer

# Output file name (%j expands to job ID)
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=12:00:00
# Allocate 1 node and 4 cores (tasks) for this job
#SBATCH --array=7,9
#SBATCH --partition=tier2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20

# Request memory (e.g., 8 GB)
#SBATCH --mem=60G

scenarios=("const" "bottleneck2000" "exp" "split500" "split1000" "split2000" "out_of_africa" "step05" "step005" "step0005")
scenario=${scenarios[$SLURM_ARRAY_TASK_ID]}

source /apps/software/gcc-12.1.0/miniconda3/24.9.2/etc/profile.d/conda.sh
source /apps/software/gcc-12.1.0/miniconda3/24.9.2/etc/profile.d/mamba.sh

mamba activate singer-snakemake-env
snakemake  --cores=20 --rerun-incomplete --configfile=../config/polegon/${scenario}_config.yaml

# Alternative config: --configfile=../config/no_polegon/${scenario}_config2.yaml
