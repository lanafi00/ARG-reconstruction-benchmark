"""
Compare SINGER-reconstructed pairwise coalescence statistics Jensen-Shannon divergence from ground truth 
across all populations in demographic scenarios,
with and without Polegon post-processing.

Assumes simulation + SINGER reconstruction were already run by a separate
script/pipeline, producing the directory layout described in `DATA_ROOT` below.
RNG seeds for the simulations themselves are tracked in that upstream script,
not here.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tskit
import tszip
from scipy.spatial.distance import jensenshannon

# ---------------------------------------------------------------------------
# Configuration / constants
#
# ---------------------------------------------------------------------------

DATA_ROOT = "official_data2"
TRUTH_TREES_DIR = f"{DATA_ROOT}/diploid_sim/trees"
PLOTS_OUTPUT_DIR = f"{DATA_ROOT}/plots_combined/jsd"
RESULTS_DIR_WITH_POLEGON = "results"
RESULTS_DIR_WITHOUT_POLEGON = "results_np"

NUM_REPLICATES = 50          # replicates per demographic scenario


# ---------------------------------------------------------------------------
# Reconstruction summarization
# ---------------------------------------------------------------------------

def _results_dir_name(use_polegon: bool) -> str:
    return RESULTS_DIR_WITH_POLEGON if use_polegon else RESULTS_DIR_WITHOUT_POLEGON


def get_reconstructed_counts(demo_name: str, time_windows: np.ndarray, use_polegon: bool):
    """
    Average pair-coalescence counts/proportions/rates across all replicate
    reconstructions of a single-population demographic scenario.
    """
    results_dir = _results_dir_name(use_polegon)
    count_list = []
    props_list = []

    for j in range(NUM_REPLICATES):
        tree_path = f"{DATA_ROOT}/{results_dir}/{demo_name}/trees/{demo_name}.{j}.tsz"
        ts = tszip.decompress(tree_path)
        count = ts.pair_coalescence_counts(time_windows=time_windows)
        count_list.append(count) 
        props_list.append(count/count.sum())

    mean_counts = np.mean(count_list, axis=0)
    mean_props = np.mean(props_list, axis=0)
    return (mean_counts, mean_props)


def get_truth_quantiles(ts: tskit.TreeSequence, time_intervals: np.ndarray):
    """Ground-truth pair-coalescence counts/proportions/rates from the true tree sequence."""
    quantiles = ts.pair_coalescence_quantiles(quantiles=time_intervals)
    return quantiles

def get_jsd(demo_name, time_intervals: np.ndarray):
    ts = tszip.decompress(f"{TRUTH_TREES_DIR}/{demo_name}.trees.tsz")
    truth_edges = get_truth_quantiles(ts, time_intervals)  # edges only

    truth_counts = ts.pair_coalescence_counts(time_windows=truth_edges)
    truth_props = truth_counts / truth_counts.sum()        

    (mean_counts_p, mean_props_p) = get_reconstructed_counts(demo_name, truth_edges, use_polegon=True)
    (mean_counts_np, mean_props_np) = get_reconstructed_counts(demo_name, truth_edges, use_polegon=False)

    js_distance_p = jensenshannon(truth_props, mean_props_p, base=2)
    js_distance_np = jensenshannon(truth_props, mean_props_np, base=2)
    js_divergence_p = js_distance_p ** 2    
    js_divergence_np = js_distance_np ** 2
    return (js_divergence_p, js_divergence_np)
    

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_jsd_bar_chart(demo_names: list, time_intervals):
    """Grouped bar chart: JSD, Polegon vs. no-Polegon, one group per demo."""
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

    jsd_polegon_list = []
    jsd_no_polegon_list = []
    for demo_name in demo_names:
        jsd_p, jsd_np = get_jsd(demo_name, time_intervals)
        jsd_polegon_list.append(jsd_p)
        jsd_no_polegon_list.append(jsd_np)
        print(f"[{demo_name}] JSD Polegon={jsd_p:.4g}, JSD no-Polegon={jsd_np:.4g}")

    x = np.arange(len(demo_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, jsd_polegon_list, width, label="SINGER + Polegon")
    ax.bar(x + width / 2, jsd_no_polegon_list, width, label="SINGER")

    ax.set_xticks(x)
    ax.set_xticklabels(demo_names, rotation=30, ha="right")
    ax.set_ylabel("Jensen-Shannon divergence")
    ax.set_title("Pairwise JSD: Polegon vs. no-Polegon\n(all samples, diploid-individual level)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_OUTPUT_DIR}/jsd_all_samples2.pdf", dpi=300, bbox_inches="tight")

 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(demo_names: list):
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    time_intervals = np.linspace(0,1,21)
    plot_jsd_bar_chart(demo_names, time_intervals)


if __name__ == "__main__":
    demos_to_run = ["const","exp","bottleneck_2000","out_of_africa","split_500","split_1000","split_2000","step_05","step_005","step_0005"]
    run(demos_to_run)
