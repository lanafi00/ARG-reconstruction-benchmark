"""
Compare ground-truth vs. SINGER-reconstructed pairwise coalescence statistics
(proportions, rates, counts) for population-comparison (split) demographic
scenarios, with and without Polegon post-processing, across all four
within-/cross-population index comparisons.

This is the specialized counterpart to the single-population script -- it
handles split_500 / split_1000 / split_2000 exclusively. Single-population
demos (const, exp, bottleneck_2000, out_of_africa, step_*) are handled there.

Assumes simulation + SINGER reconstruction were already run by a separate
script/pipeline, producing the directory layout described in `DATA_ROOT` below.
RNG seeds for the simulations themselves are tracked in that upstream script,
not here.
"""

import os
from dataclasses import dataclass

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
PLOTS_OUTPUT_DIR = f"{DATA_ROOT}/plots_combined"
RESULTS_DIR_WITH_POLEGON = "results"
RESULTS_DIR_WITHOUT_POLEGON = "results_np"

NUM_REPLICATES = 50          # replicates per demographic scenario
N_TIME_BINS = 20             # number of log-spaced time bins for pair coalescence stats

# Population metadata name tags used in these split tree sequences.
# NOTE: the ground-truth simulator and the SINGER reconstruction pipeline
# label populations differently -- this is expected, not a bug, and both
# must be kept in sync with however each pipeline currently names things.
TRUTH_POP_A_METADATA_NAME = "popA"
TRUTH_POP_B_METADATA_NAME = "popB"
RECONSTRUCTION_POP_A_METADATA_NAME = "DpopA"
RECONSTRUCTION_POP_B_METADATA_NAME = "DpopB"

# The four within-/cross-population index comparisons evaluated for every
# split demo: (0,0) and (1,1) are within-population, (0,1)/(1,0) cross-population.
POP_COMPARISON_INDEXES = [(0, 0), (0, 1), (1, 1)]


# ---------------------------------------------------------------------------
# Analytic (closed-form) coalescence rate function for split demographies
#
# Parameters are hardcoded to match whatever was used to generate the
# simulations in `TRUTH_TREES_DIR` -- if the upstream simulation parameters
# change, these must be updated to match, or the "Empirical Ne(t)" curve will
# silently be wrong.
# ---------------------------------------------------------------------------

def coal_rate_split(t: float, split_time: float, pop_comparison: tuple,
                     split_N: float = 5_000, before_N: float = 10_000) -> float:
    """
    pop_comparison: index pair identifying which populations are being
    compared, e.g. (0, 1) for cross-population coalescence, (0, 0)/(1, 1) for
    within-population.
    """
    is_cross_population = pop_comparison in [(0, 1), (1, 0)]
    if is_cross_population:
        return 0 if t < split_time else 1 / (2 * before_N)
    else:
        return 1 / (2 * split_N) if t < split_time else 1 / (2 * before_N)


# ---------------------------------------------------------------------------
# Demographic scenario registry
#
# Single source of truth for the split time associated with each split demo.
# To add a new split demo: add one entry here.
# ---------------------------------------------------------------------------

@dataclass
class SplitDemoConfig:
    name: str
    split_time: float


DEMOS: dict[str, SplitDemoConfig] = {
    "split_500":  SplitDemoConfig("split_500", split_time=500),
    "split_1000": SplitDemoConfig("split_1000", split_time=1000),
    "split_2000": SplitDemoConfig("split_2000", split_time=2000),
}


# ---------------------------------------------------------------------------
# Reconstruction / truth summarization
# ---------------------------------------------------------------------------

def _results_dir_name(use_polegon: bool) -> str:
    return RESULTS_DIR_WITH_POLEGON if use_polegon else RESULTS_DIR_WITHOUT_POLEGON


def _get_named_populations(ts: tskit.TreeSequence, pop_a_name: str, pop_b_name: str):
    """Return [popA, popB] tskit Population objects matched by metadata name, or None for missing."""
    pops = [None, None]
    for pop in ts.populations():
        metadata = pop.metadata
        if metadata and "name" in metadata:
            if metadata["name"] == pop_a_name:
                pops[0] = pop
            elif metadata["name"] == pop_b_name:
                pops[1] = pop
    return pops


def get_reconstructed_means_pop_comparison(pop_indexes, demo_name: str,
                                            time_intervals: np.ndarray, use_polegon: bool):
    """
    Average pair-coalescence counts/proportions/rates across all replicate
    reconstructions of a split demographic scenario, for one population
    index comparison (e.g. within-pop-A, or cross-population).
    """
    results_dir = _results_dir_name(use_polegon)
    counts_list, props_list, rate_list = [], [], []
    n_skipped = 0

    for j in range(NUM_REPLICATES):
        tree_path = f"{DATA_ROOT}/{results_dir}/{demo_name}/trees/{demo_name}.{j}.tsz"
        ts = tszip.decompress(tree_path)

        pops = _get_named_populations(ts, RECONSTRUCTION_POP_A_METADATA_NAME,
                                       RECONSTRUCTION_POP_B_METADATA_NAME)
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
    mean_rate = hmean(np.array(rate_list), axis=0)
    return mean_counts, mean_props, mean_rate


def get_truth_pop_comparison(pop_indexes, ts: tskit.TreeSequence, time_intervals: np.ndarray):
    """Ground-truth pair-coalescence counts/proportions/rates for one population index comparison."""
    pops = _get_named_populations(ts, TRUTH_POP_A_METADATA_NAME, TRUTH_POP_B_METADATA_NAME)
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


def plot_proportions(demo_name, pop_comparison, plot_intervals,
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
              f"population comparison {pop_comparison}\nDemographic model: {demo_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_OUTPUT_DIR}/{demo_name}_props{pop_comparison}.pdf", dpi=300, bbox_inches="tight")


def plot_rates(demo_name, pop_comparison, split_time, plot_intervals, eval_times, max_time,
               truth_rate, polegon_rate, no_polegon_rate):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs(truth_rate[1:-1], plot_intervals[1:-1], color="blue", linewidth=1.5, label="Truth")
    ax.stairs(polegon_rate[1:-1], plot_intervals[1:-1], color="red", linewidth=1.5, label="SINGER + Polegon")
    ax.stairs(no_polegon_rate[1:-1], plot_intervals[1:-1], color="green", linewidth=1.5, label="SINGER")

    empirical_rates = [coal_rate_split(t, split_time, pop_comparison) if t <= max_time else np.nan
                        for t in eval_times]
    ax.stairs(empirical_rates, plot_intervals[1:], color="black", linewidth=1.5,
              linestyle="--", label="Empirical Ne(t)")

    ax.set_xscale("log")
    ax.set_xlabel("Generations in past")
    ax.set_ylabel("Pair coalescent rates")
    ax.legend()
    plt.title(f"Ground truth + SINGER reconstruction coalescent rates over time, "
              f"population comparison {pop_comparison}\nDemographic model: {demo_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_OUTPUT_DIR}/{demo_name}_rates{pop_comparison}.pdf", dpi=300, bbox_inches="tight")
  


def plot_counts(demo_name, pop_comparison, plot_intervals,
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
              f"population comparison {pop_comparison}\nDemographic model: {demo_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_OUTPUT_DIR}/{demo_name}_counts{pop_comparison}.pdf", dpi=300, bbox_inches="tight")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(demo_name: str):
    if demo_name not in DEMOS:
        raise ValueError(
            f"'{demo_name}' is not a recognized split/population-comparison demo. "
            f"Known demos: {sorted(DEMOS)}"
        )
    config = DEMOS[demo_name]

    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

    mutated = tszip.decompress(f"{TRUTH_TREES_DIR}/{demo_name}.trees.tsz")
    max_time = mutated.max_time
    time_intervals, plot_intervals = make_time_intervals(max_time)
    eval_times = plot_intervals[1:-1]

    for pop_comparison in POP_COMPARISON_INDEXES:
        pop_indexes = [pop_comparison]

        polegon_counts, polegon_props, polegon_rate = get_reconstructed_means_pop_comparison(
            pop_indexes, demo_name, time_intervals, use_polegon=True
        )
        no_polegon_counts, no_polegon_props, no_polegon_rate = get_reconstructed_means_pop_comparison(
            pop_indexes, demo_name, time_intervals, use_polegon=False
        )
        truth_counts, truth_props, truth_rate = get_truth_pop_comparison(
            pop_indexes, mutated, time_intervals
        )

        plot_proportions(demo_name, pop_comparison, plot_intervals,
                          truth_props, polegon_props, no_polegon_props)
        plot_rates(demo_name, pop_comparison, config.split_time, plot_intervals, eval_times, max_time,
                   truth_rate, polegon_rate, no_polegon_rate)
        plot_counts(demo_name, pop_comparison, plot_intervals,
                    truth_counts, polegon_counts, no_polegon_counts)


if __name__ == "__main__":
    demos_to_run = ["split_500", "split_1000", "split_2000"]

    for demo_name in demos_to_run:
        run(demo_name)
