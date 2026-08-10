"""
Compare ground-truth vs. SINGER-reconstructed pairwise coalescence statistics
(proportions, rates, counts) across single-population demographic scenarios,
with and without Polegon post-processing.

NOTE: population comparisons for population-comparison demos (split_500, split_1000, split_2000, and any
stepping-stone/multi-population models) are handled by a separate specialized
script and are intentionally excluded here.

Assumes simulation + SINGER reconstruction were already run by a separate
script/pipeline, producing the directory layout described in `DATA_ROOT` below.
RNG seeds for the simulations themselves are tracked in that upstream script,
not here.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
import tskit
import tszip
from scipy.stats import hmean

# ---------------------------------------------------------------------------
# Configuration / constants
#
# These describe assumptions baked into the upstream simulation + SINGER
# pipeline. If that pipeline changes (replicate count, directory names),
# update here to match.
# ---------------------------------------------------------------------------

DATA_ROOT = "official_data2"
TRUTH_TREES_DIR = f"{DATA_ROOT}/diploid_sim/trees"
PLOTS_OUTPUT_DIR = f"{DATA_ROOT}/plots_combined/quantiles"
RESULTS_DIR_WITH_POLEGON = "results"
RESULTS_DIR_WITHOUT_POLEGON = "results_np"

NUM_REPLICATES = 50          # replicates per demographic scenario
N_TIME_BINS = 20             # number of log-spaced time bins for pair coalescence stats


# ---------------------------------------------------------------------------
# Analytic (closed-form) coalescence rate functions
#
# Each function returns the pairwise coalescence rate 1/(2N(t)) implied by the
# demographic model's known parameters. These parameters are hardcoded to
# match whatever was used to generate the simulations in `TRUTH_TREES_DIR`
# -- if the upstream simulation parameters change, these must be updated
# to match, or the "Empirical Ne(t)" curve will silently be wrong.
# ---------------------------------------------------------------------------

def coal_rate_const(t: float, N: float = 10_000) -> float:
    return 1.0 / (2 * N)


def coal_rate_bottleneck(t: float, N_recent: float = 5_000,
                          N_ancestral: float = 10_000,
                          t_change: float = 2_000) -> float:
    N = N_recent if t < t_change else N_ancestral
    return 1.0 / (2 * N)


def coal_rate_exp(t: float, start_N: float = 10_000,
                   growth_rate: float = np.log(4) / 250,
                   start_exp_time: float = 250) -> float:
    if t < start_exp_time:
        N = start_N * np.exp(-growth_rate * t)
    else:
        N = start_N * np.exp(-growth_rate * start_exp_time)
    return 1.0 / (2 * N)


# ---------------------------------------------------------------------------
# Demographic scenario registry
#
# Single source of truth for which single-population demos exist, and
# whether they have a closed-form analytic rate to overlay. To add a new demographic scenario: add one entry here. No other
# function needs to change.
# ---------------------------------------------------------------------------

@dataclass
class DemoConfig:
    name: str
    analytic_fn: Optional[Callable[[float], float]] = None  # None => no closed-form Ne(t) overlay


DEMOS: dict[str, DemoConfig] = {
    "const":           DemoConfig("const", analytic_fn=coal_rate_const),
    "exp":             DemoConfig("exp", analytic_fn=coal_rate_exp),
    "out_of_africa":   DemoConfig("out_of_africa"),
    "bottleneck_2000": DemoConfig("bottleneck_2000", analytic_fn=coal_rate_bottleneck),
    "step_05":         DemoConfig("step_05"),
    "step_005":        DemoConfig("step_005"),
    "step_0005":       DemoConfig("step_0005"),
    "split_500":       DemoConfig("split_500"),
    "split_1000":      DemoConfig("split_1000"),
    "split_2000":      DemoConfig("split_2000"),
}

# Which demos get the black dashed "Empirical Ne(t)" overlay in the rates plot.
ANALYTIC_OVERLAY_DEMOS = {"const", "exp", "bottleneck_2000"}


# ---------------------------------------------------------------------------
# Reconstruction summarization
# ---------------------------------------------------------------------------

def _results_dir_name(use_polegon: bool) -> str:
    return RESULTS_DIR_WITH_POLEGON if use_polegon else RESULTS_DIR_WITHOUT_POLEGON


def get_reconstructed_means(demo_name: str, time_intervals: np.ndarray, use_polegon: bool):
    """
    Average pair-coalescence counts/proportions/rates across all replicate
    reconstructions of a single-population demographic scenario.
    """
    results_dir = _results_dir_name(use_polegon)
    counts_list, props_list, rate_list = [], [], []

    for j in range(NUM_REPLICATES):
        tree_path = f"{DATA_ROOT}/{results_dir}/{demo_name}/trees/{demo_name}.{j}.tsz"
        ts = tszip.decompress(tree_path)

        count = ts.pair_coalescence_counts(time_windows=time_intervals, pair_normalise=False)
        reconstruction_props = count / count.sum()
        reconstruction_rate = ts.pair_coalescence_rates(time_windows=time_intervals)

        counts_list.append(count)
        props_list.append(reconstruction_props)
        rate_list.append(reconstruction_rate)

    mean_counts = np.mean(counts_list, axis=0)
    mean_props = np.mean(props_list, axis=0)
    mean_rate = hmean(np.array(rate_list), axis=0)
    return mean_counts, mean_props, mean_rate


def get_truth_stats(ts: tskit.TreeSequence, time_intervals: np.ndarray):
    """Ground-truth pair-coalescence counts/proportions/rates from the true tree sequence."""
    counts = ts.pair_coalescence_counts(time_windows=time_intervals, pair_normalise=False)
    proportions = counts / counts.sum()
    rates = ts.pair_coalescence_rates(time_windows=time_intervals)
    return counts, proportions, rates

def get_truth_quantiles(ts: tskit.TreeSequence, time_intervals: np.ndarray):
    """Ground-truth pair-coalescence counts/proportions/rates from the true tree sequence."""
    quantiles = ts.pair_coalescence_quantiles(quantiles=time_intervals)
    return quantiles


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


def plot_proportions(demo_name, plot_intervals, truth_props, polegon_props, no_polegon_props):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs(truth_props[1:-1], plot_intervals[1:-1], color="blue", linewidth=1.5, label="Truth")
    ax.stairs(polegon_props[1:-1], plot_intervals[1:-1], color="red", linewidth=1.5, label="SINGER + Polegon")
    ax.stairs(no_polegon_props[1:-1], plot_intervals[1:-1], color="green", linewidth=1.5, label="SINGER")
    ax.set_xscale("log")
    ax.set_xlabel("Generations in past")
    ax.set_ylabel("Proportion coalescing pairs")
    ax.legend()
    plt.title(f"Ground truth + SINGER reconstruction proportion coalescing pairs over time\n"
              f"Demographic model: {demo_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_OUTPUT_DIR}/{demo_name}_props.pdf", dpi=300, bbox_inches="tight")
    plt.show()


def plot_rates(demo_name, plot_intervals, eval_times, max_time,
                truth_rate, polegon_rate, no_polegon_rate):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs(truth_rate[1:-1], plot_intervals[1:-1], linewidth=1.5, label="Truth")
    ax.stairs(polegon_rate[1:-1], plot_intervals[1:-1], linewidth=1.5, label="SINGER + Polegon")
    ax.stairs(no_polegon_rate[1:-1], plot_intervals[1:-1], linewidth=1.5, label="SINGER")

    if demo_name in ANALYTIC_OVERLAY_DEMOS:
        analytic_fn = DEMOS[demo_name].analytic_fn
        empirical_rates = [analytic_fn(t) if t <= max_time else np.nan for t in eval_times]
        ax.stairs(empirical_rates, plot_intervals[1:], color="black", linewidth=1.5,
                  linestyle="--", label="Empirical Ne(t)")

    ax.set_xscale("log")
    ax.set_xlabel("Generations in past")
    ax.set_ylabel("Pair coalescent rates")
    ax.legend()
    plt.title(f"Ground truth + SINGER reconstruction coalescent rates over time\n"
              f"Demographic model: {demo_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_OUTPUT_DIR}/{demo_name}_rates.pdf", dpi=300, bbox_inches="tight")
    plt.show()


def plot_counts(demo_name, plot_intervals, truth_counts, polegon_counts, no_polegon_counts):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs(truth_counts[1:-1], plot_intervals[1:-1], linewidth=1.5, label="Truth")
    ax.stairs(polegon_counts[1:-1], plot_intervals[1:-1], linewidth=1.5, label="SINGER + Polegon")
    ax.stairs(no_polegon_counts[1:-1], plot_intervals[1:-1], linewidth=1.5, label="SINGER")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)  # avoid log(0): clips zero/sub-1 counts to the bottom edge
    ax.set_xlabel("Generations in past")
    ax.set_ylabel("Number coalescing pairs")
    ax.legend()
    plt.title(f"Ground truth + SINGER reconstruction number of coalescing pairs over time\n"
              f"Demographic model: {demo_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_OUTPUT_DIR}/{demo_name}_counts.pdf", dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(demo_name: str):
    if demo_name not in DEMOS:
        raise ValueError(
            f"'{demo_name}' is not a recognized demo. "
            f"If it's a split/population-comparison demo, use the specialized script instead. "
            f"Known demos: {sorted(DEMOS)}"
        )

    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

    mutated = tszip.decompress(f"{TRUTH_TREES_DIR}/{demo_name}.trees.tsz")
    max_time = mutated.max_time
    time_intervals = np.linspace(0,1,21)
    #Alternatively you can use make_time_intervals() if you don't want a quantiles-based plot
    plot_intervals = get_truth_quantiles(mutated, time_intervals)  # edges only
    plot_intervals = np.concatenate(([0], plot_intervals, [np.inf]))
    eval_times = plot_intervals[1:-1]

    polegon_counts, polegon_props, polegon_rate = get_reconstructed_means(
        demo_name, plot_intervals, use_polegon=True
    )
    no_polegon_counts, no_polegon_props, no_polegon_rate = get_reconstructed_means(
        demo_name, plot_intervals, use_polegon=False
    )
    truth_counts, truth_props, truth_rate = get_truth_stats(mutated, plot_intervals)

    plot_proportions(demo_name, plot_intervals, truth_props, polegon_props, no_polegon_props)
    plot_rates(demo_name, plot_intervals, eval_times, max_time,
                truth_rate, polegon_rate, no_polegon_rate)
    plot_counts(demo_name, plot_intervals, truth_counts, polegon_counts, no_polegon_counts)


if __name__ == "__main__":
    demos_to_run = ["split_500","const","exp","out_of_africa","bottleneck_2000","split_1000","split_2000","step_05","step_0005"]
    for demo_name in demos_to_run:
        run(demo_name)
