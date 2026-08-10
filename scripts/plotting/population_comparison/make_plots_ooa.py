"""
Compare ground-truth vs. SINGER-reconstructed pairwise coalescence statistics
(proportions, rates, counts) for the out-of-Africa (YRI/CEU/CHB) demographic
model, with and without Polegon post-processing, across all five within-/
cross-population index comparisons.

No closed-form Ne(t) is known for this model, so unlike the split-demo script,
no analytic overlay is drawn on the rates plot.

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
from scipy.stats import hmean

# ---------------------------------------------------------------------------
# Configuration / constants
#
# These describe assumptions baked into the upstream simulation + SINGER
# pipeline. If that pipeline changes (replicate count, directory names,
# population metadata labels), update here to match.
# ---------------------------------------------------------------------------

DATA_ROOT = "official_data2"
TRUTH_TREES_DIR = f"{DATA_ROOT}/diploid_sim/trees"
PLOTS_OUTPUT_DIR = f"{DATA_ROOT}/plots"
RESULTS_DIR_WITH_POLEGON = "results"
RESULTS_DIR_WITHOUT_POLEGON = "results_np"

NUM_REPLICATES = 50          # replicates per demographic scenario
N_TIME_BINS = 20             # number of log-spaced time bins for pair coalescence stats

# Population metadata name tags used in these out-of-Africa tree sequences.
# NOTE: truth and reconstruction happen to use the same names here (unlike
# the split-demo pipeline, where they differ) -- kept as separate constants
# anyway so a future naming mismatch is a one-line fix, not a search-and-replace.
TRUTH_POP_A_METADATA_NAME = "YRI"
TRUTH_POP_B_METADATA_NAME = "CEU"
TRUTH_POP_C_METADATA_NAME = "CHB"
RECONSTRUCTION_POP_A_METADATA_NAME = "DYRI"
RECONSTRUCTION_POP_B_METADATA_NAME = "DCEU"
RECONSTRUCTION_POP_C_METADATA_NAME = "DCHB"

# The five within-/cross-population index comparisons evaluated: (0,0), (1,1),
# (2,2) are within-population (YRI-YRI, CEU-CEU, CHB-CHB); (0,1) and (1,2) are
# cross-population (YRI-CEU, CEU-CHB). Indexes refer to sample_sets order
# [popA, popB, popC] = [YRI, CEU, CHB].
POP_COMPARISON_INDEXES = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
POP_COMPARISON_LABELS = {
    (0, 0): "(YRI,YRI)",
    (0, 1): "(YRI,CEU)",
    (1, 1): "(CEU,CEU)",
    (1, 2): "(CEU,CHB)",
    (2, 2): "(CHB,CHB)",
}


# ---------------------------------------------------------------------------
# Reconstruction / truth summarization
# ---------------------------------------------------------------------------

def _results_dir_name(use_polegon: bool) -> str:
    return RESULTS_DIR_WITH_POLEGON if use_polegon else RESULTS_DIR_WITHOUT_POLEGON


def _get_named_populations(ts: tskit.TreeSequence, pop_a_name: str, pop_b_name: str, pop_c_name: str):
    """Return [popA, popB, popC] tskit Population objects matched by metadata name, or None for missing."""
    pops = [None, None, None]
    for pop in ts.populations():
        metadata = pop.metadata
        if metadata and "name" in metadata:
            if metadata["name"] == pop_a_name:
                pops[0] = pop
            elif metadata["name"] == pop_b_name:
                pops[1] = pop
            elif metadata["name"] == pop_c_name:
                pops[2] = pop
    return pops


def get_reconstructed_means_pop_comparison(pop_indexes, demo_name: str,
                                            time_intervals: np.ndarray, use_polegon: bool):
    """
    Average pair-coalescence counts/proportions/rates across all replicate
    reconstructions, for one population index comparison (e.g. within-YRI,
    or YRI-CEU cross-population).
    """
    results_dir = _results_dir_name(use_polegon)
    counts_list, props_list, rate_list = [], [], []
    n_skipped = 0

    for j in range(NUM_REPLICATES):
        tree_path = f"{DATA_ROOT}/{results_dir}/{demo_name}/trees/{demo_name}.{j}.tsz"
        ts = tszip.decompress(tree_path)

        pops = _get_named_populations(ts, RECONSTRUCTION_POP_A_METADATA_NAME,
                                       RECONSTRUCTION_POP_B_METADATA_NAME,
                                       RECONSTRUCTION_POP_C_METADATA_NAME)
        if None in pops:
            print(f"[{demo_name}] warning: missing population in tree {j}, skipping")
            n_skipped += 1
            continue

        raw_sample_sets = [list(ts.samples(population=pop.id)) for pop in pops]
        sample_sets = [s for s in raw_sample_sets if len(s) > 0]  # drop empty populations

        reconstruction_rate = ts.pair_coalescence_rates(
            time_windows=time_intervals, sample_sets=sample_sets, indexes=pop_indexes
        )
        count = ts.pair_coalescence_counts(
            time_windows=time_intervals, sample_sets=sample_sets,
            indexes=pop_indexes, pair_normalise=False
        )
        counts_list.append(count[0])
        reconstruction_props = count[0] / count[0].sum()
        props_list.append(reconstruction_props)
        assert len(reconstruction_props) == len(time_intervals) - 1
        rate_list.append(reconstruction_rate[0])

    if n_skipped:
        print(f"[{demo_name}] used {NUM_REPLICATES - n_skipped}/{NUM_REPLICATES} replicates "
              f"({n_skipped} skipped due to missing populations)")

    mean_counts = np.mean(counts_list, axis=0)
    mean_props = np.mean(props_list, axis=0)
    mean_rate = hmean(rate_list, axis=0)
    return mean_counts, mean_props, mean_rate


def get_truth_pop_comparison(pop_indexes, ts: tskit.TreeSequence, time_intervals: np.ndarray):
    """Ground-truth pair-coalescence counts/proportions/rates for one population index comparison."""
    pops = _get_named_populations(ts, TRUTH_POP_A_METADATA_NAME, TRUTH_POP_B_METADATA_NAME,
                                   TRUTH_POP_C_METADATA_NAME)
    if None in pops:
        print("warning: missing population in ground-truth tree")
        return None, None, None

    sample_sets = [list(ts.samples(population=pop.id)) for pop in pops]
    counts = ts.pair_coalescence_counts(
        time_windows=time_intervals, sample_sets=sample_sets,
        indexes=pop_indexes, pair_normalise=False
    )[0]
    proportions = counts / counts.sum()
    rates = ts.pair_coalescence_rates(
        time_windows=time_intervals, sample_sets=sample_sets, indexes=pop_indexes
    )[0]
    return counts, proportions, rates


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_time_intervals(max_time: float):
    """
    Build log-spaced time bin edges for pair-coalescence statistics, plus a
    version suitable for plotting (with a finite, extrapolated final edge
    instead of np.inf).
    """
    time_intervals = np.logspace(0, np.log10(max_time), N_TIME_BINS)
    time_intervals = np.concatenate(([0], time_intervals, [np.inf]))

    plot_intervals = time_intervals.copy()
    # Replace the infinite final edge with an extrapolated finite value so
    # ax.stairs has something plottable for the last bin.
    plot_intervals[-1] = plot_intervals[-2] * (plot_intervals[-2] / plot_intervals[-3])

    return time_intervals, plot_intervals


def plot_proportions(demo_name, pop_comparison_label, plot_intervals,
                      truth_props, polegon_props, no_polegon_props):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs(truth_props[1:], plot_intervals[1:], color="blue", linewidth=1.5, label="Truth")
    ax.stairs(polegon_props[1:], plot_intervals[1:], color="red", linewidth=1.5, label="SINGER + Polegon")
    ax.stairs(no_polegon_props[1:], plot_intervals[1:], color="green", linewidth=1.5, label="SINGER")
    ax.set_xscale("log")
    ax.set_xlabel("Generations in past")
    ax.set_ylabel("Proportion coalescing pairs")
    ax.legend()
    plt.title(f"Ground truth + SINGER reconstruction proportion coalescing pairs over time, "
              f"population comparison {pop_comparison_label}\nDemographic model: {demo_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_OUTPUT_DIR}/{demo_name}_props{pop_comparison_label}.pdf", dpi=300, bbox_inches="tight")
    plt.show()


def plot_rates(demo_name, pop_comparison_label, plot_intervals,
               truth_rate, polegon_rate, no_polegon_rate):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs(truth_rate[1:-1], plot_intervals[1:-1], color="blue", linewidth=1.5, label="Truth")
    ax.stairs(polegon_rate[1:-1], plot_intervals[1:-1], color="red", linewidth=1.5, label="SINGER + Polegon")
    ax.stairs(no_polegon_rate[1:-1], plot_intervals[1:-1], color="green", linewidth=1.5, label="SINGER")
    # No closed-form Ne(t) for out-of-Africa, so no "Empirical Ne(t)" overlay here.
    ax.set_xscale("log")
    ax.set_xlabel("Generations in past")
    ax.set_ylabel("Pair coalescent rates")
    ax.legend()
    plt.title(f"Ground truth + SINGER reconstruction coalescent rates over time, "
              f"population comparison {pop_comparison_label}\nDemographic model: {demo_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_OUTPUT_DIR}/{demo_name}_rates{pop_comparison_label}.pdf", dpi=300, bbox_inches="tight")
    plt.show()


def plot_counts(demo_name, pop_comparison_label, plot_intervals,
                 truth_counts, polegon_counts, no_polegon_counts):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs(truth_counts[1:], plot_intervals[1:], color="blue", linewidth=1.5, label="Truth")
    ax.stairs(polegon_counts[1:], plot_intervals[1:], color="red", linewidth=1.5, label="SINGER + Polegon")
    ax.stairs(no_polegon_counts[1:], plot_intervals[1:], color="green", linewidth=1.5, label="SINGER")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)  # avoid log(0): clips zero/sub-1 counts to the bottom edge
    ax.set_xlabel("Generations in past")
    ax.set_ylabel("Number coalescing pairs")
    ax.legend()
    plt.title(f"Ground truth + SINGER reconstruction number of coalescing pairs over time, "
              f"population comparison {pop_comparison_label}\nDemographic model: {demo_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_OUTPUT_DIR}/{demo_name}_counts{pop_comparison_label}.pdf", dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(demo_name: str):
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

    mutated = tszip.decompress(f"{TRUTH_TREES_DIR}/{demo_name}.trees.tsz")
    max_time = mutated.max_time
    time_intervals, plot_intervals = make_time_intervals(max_time)

    for pop_comparison in POP_COMPARISON_INDEXES:
        pop_indexes = [pop_comparison]
        pop_comparison_label = POP_COMPARISON_LABELS[pop_comparison]

        polegon_counts, polegon_props, polegon_rate = get_reconstructed_means_pop_comparison(
            pop_indexes, demo_name, time_intervals, use_polegon=True
        )
        no_polegon_counts, no_polegon_props, no_polegon_rate = get_reconstructed_means_pop_comparison(
            pop_indexes, demo_name, time_intervals, use_polegon=False
        )
        truth_counts, truth_props, truth_rate = get_truth_pop_comparison(
            pop_indexes, mutated, time_intervals
        )

        plot_proportions(demo_name, pop_comparison_label, plot_intervals,
                          truth_props, polegon_props, no_polegon_props)
        plot_rates(demo_name, pop_comparison_label, plot_intervals,
                   truth_rate, polegon_rate, no_polegon_rate)
        plot_counts(demo_name, pop_comparison_label, plot_intervals,
                    truth_counts, polegon_counts, no_polegon_counts)


if __name__ == "__main__":
    demos_to_run = ["out_of_africa"]

    for demo_name in demos_to_run:
        run(demo_name)
