# Benchmarking SINGER ARG Reconstruction 
 I benchmark SINGER ([`singer-snakemake`](https://github.com/nspope/singer-snakemake)) across ten different demographic models, ranging from one to sixteen subpopulations and including size changes such as bottlenecks and exponential growth. Each model is simulated using the msprime engine. For each scenario, SINGER reconstructs 50 plausible tree sequences from the simulated VCF output. Pairwise coalescent rates are used to compare these reconstructed tree sequences against the ground-truth tree sequence. 


## Setup

This project builds on [`singer-snakemake`](https://github.com/nspope/singer-snakemake), 
a Snakemake workflow for running SINGER. That tool is not included in this 
repo. Clone it separately, then use the configs and scripts here on top of it.

1. Clone the `singer-snakemake` tool:
```bash
   git clone https://github.com/nspope/singer-snakemake
   cd singer-snakemake
   mamba env create -f environment.yaml
   mamba activate singer-snakemake-env
```

2. Clone this repo (for configs and analysis/plotting scripts):
```bash
   git clone https://github.com/lanafi00/NITMB_ARG_Inference.git
```

## Usage

1. **Simulate demographic models** — generate ground-truth tree sequences, 
   VCFs, and population metadata for each of the ten scenarios (constant, bottleneck, 
   exponential, three split-population variants, two stepping-stone variants, and the Out-of-Africa 3G09 model from the Stdpopsim catalogue,):
```bash
   python scripts/official_sim_gen2.py <j>
   python scripts/write_population_metadata.py <j>
```
   `j` is a job index (0–9, one per demographic scenario) read directly from 
   `sys.argv[1]`. To run the full set on the cluster as a SLURM array (adjusting source and file paths as necessary):

```bash
   sbatch --array=0-9 cluster/run_simulation.sl
```

2. **Run SINGER** — from inside the `singer-snakemake` directory, point Snakemake 
   at the relevant config from this repo (optionally using Polegon):
```bash
   snakemake --cores=20 --configfile=<path-to>/ARG-reconstruction-benchmark/config/polegon/<model>_config.yaml
```
```bash
   snakemake --cores=20 --configfile=<path-to>/ARG-reconstruction-benchmark/config/no_polegon/<model>_config.yaml
```
   To run SINGER on all models on the cluster as a SLURM array (adjusting source and file paths as necessary):

```bash
   sbatch --array=0-9 <path-to>/ARG-reconstruction-benchmark/cluster/singer_submission.sl
```

3. **Plot ground truth vs reconstructions** — for each model, compare the reconstructed tree sequences 
   against the ground truth tree sequence by plotting pairwise coalescence rates, counts, and proportions:

```bash
   python plotting/polegon/make_plots4.py
```
  Model-specific plotting scripts create additional plots comparing within-population and between-population statistics.
  
```bash
   python plotting/polegon/population_comparison/make_plots_ooa.py
   python plotting/polegon/population_comparison/make_plots_step.py
   python plotting/polegon/population_comparison/make_plots_split.py
```
   Adjust file paths to SINGER's output as necessary. 
 
   
## Repository structure
- `config` — per-scenario SINGER configs (used with the `singer-snakemake` tool)
- `scripts/` — simulation and plotting code
- `cluster/` — SLURM submission scripts for the CRI cluster
