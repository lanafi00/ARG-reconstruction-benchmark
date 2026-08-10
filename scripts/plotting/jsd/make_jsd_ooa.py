import os
import numpy as np
import matplotlib.pyplot as plt
import tszip
from scipy.spatial.distance import jensenshannon

DATA_ROOT = "official_data2"
TRUTH_TREES_DIR = f"{DATA_ROOT}/diploid_sim/trees"
PLOTS_OUTPUT_DIR = f"{DATA_ROOT}/plots_combined/jsd"
RESULTS_DIR_WITH_POLEGON = "results"
RESULTS_DIR_WITHOUT_POLEGON = "results_np"

NUM_REPLICATES = 50
QUANTILE_PROBS = np.linspace(0, 1, 21)  # 21 edges -> 20 bins, ~5% truth mass each

DEMO_NAME = "step_005"
TRUTH_NAMES = ["pop_1", "pop_9", "pop_73", "pop_81", "pop_41"]
RECON_NAMES = ["D0_D0", "D0_D8", "D8_D0", "D8_D8", "D4_D4"]

# Human-readable labels for each deme, used to build comparison labels below.
# Based on the reconstruction pipeline's grid-coordinate naming.
DEME_LABELS = ["D0_D0", "D0_D8", "D8_D0", "D8_D8", "D4_D4"]

# All within-/cross-deme index comparisons: 5 within-deme (0,0)...(4,4) plus
# all 10 cross-deme pairs, 15 total.
COMPARISON_INDEXES = [(0,0),(4,4),(0,4),(0,1),(0,3)]
COMPARISON_LABELS = {
    (i, j): f"({DEME_LABELS[i]},{DEME_LABELS[j]})"
    for (i, j) in COMPARISON_INDEXES
}


def results_dir(use_polegon):
    return RESULTS_DIR_WITH_POLEGON if use_polegon else RESULTS_DIR_WITHOUT_POLEGON


def get_named_sample_sets(ts, names):
    """Returns sample-node lists in `names` order, matched by population
    metadata name, or None if any name is missing from this tree."""
    pops = [None] * len(names)
    for pop in ts.populations():
        metadata = pop.metadata
        if metadata and "name" in metadata and metadata["name"] in names:
            pops[names.index(metadata["name"])] = pop
    if None in pops:
        return None
    return [list(ts.samples(population=pop.id)) for pop in pops]


def jsd(truth_props, recon_props):
    return jensenshannon(truth_props, recon_props, base=2) ** 2


# ---------------------------------------------------------------------------
# Ground truth, loaded once
# ---------------------------------------------------------------------------

truth_ts = tszip.decompress(f"{TRUTH_TREES_DIR}/{DEMO_NAME}.trees.tsz")
truth_sample_sets = get_named_sample_sets(truth_ts, TRUTH_NAMES)
if truth_sample_sets is None:
    raise RuntimeError(f"Could not find all of {TRUTH_NAMES} in the ground-truth tree.")

results = {}  # label -> (jsd_polegon, jsd_no_polegon)

for pop_comparison in COMPARISON_INDEXES:
    indexes = [pop_comparison]
    label = COMPARISON_LABELS[pop_comparison]

    edges = truth_ts.pair_coalescence_quantiles(
        quantiles=QUANTILE_PROBS, sample_sets=truth_sample_sets, indexes=indexes
    )[0]
    truth_counts = truth_ts.pair_coalescence_counts(
        time_windows=edges, sample_sets=truth_sample_sets, indexes=indexes, pair_normalise=False
    )[0]
    truth_props = truth_counts / truth_counts.sum()

    jsd_values = {}
    for use_polegon in (True, False):
        props_list = []
        n_skipped = 0
        for j in range(NUM_REPLICATES):
            tree_path = f"{DATA_ROOT}/{results_dir(use_polegon)}/{DEMO_NAME}/trees/{DEMO_NAME}.{j}.tsz"
            ts = tszip.decompress(tree_path)
            sample_sets = get_named_sample_sets(ts, RECON_NAMES)
            if sample_sets is None:
                n_skipped += 1
                continue
            counts = ts.pair_coalescence_counts(
                time_windows=edges, sample_sets=sample_sets, indexes=indexes, pair_normalise=False
            )[0]
            props_list.append(counts / counts.sum())
        if n_skipped:
            print(f"[{label}] {'Polegon' if use_polegon else 'no-Polegon'}: "
                  f"skipped {n_skipped}/{NUM_REPLICATES} replicates (missing population)")
        mean_props = np.mean(props_list, axis=0)
        jsd_values[use_polegon] = jsd(truth_props, mean_props)

    results[label] = (jsd_values[True], jsd_values[False])
    print(f"[{label}] JSD Polegon={results[label][0]:.4g}, JSD no-Polegon={results[label][1]:.4g}")


# ---------------------------------------------------------------------------
# Plot: one figure, comparisons listed along the y-axis
# ---------------------------------------------------------------------------

os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

labels = list(results.keys())
jsd_polegon = [results[l][0] for l in labels]
jsd_no_polegon = [results[l][1] for l in labels]

y = np.arange(len(labels))
height = 0.35

fig, ax = plt.subplots(figsize=(8, 0.6 * len(labels) + 2))
ax.barh(y - height / 2, jsd_polegon, height, label="SINGER + Polegon")
ax.barh(y + height / 2, jsd_no_polegon, height, label="SINGER")

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("Jensen-Shannon divergence")
ax.set_title(f"Pairwise coalescence JSD: Polegon vs. no-Polegon\nDemographic model: {DEMO_NAME}")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_OUTPUT_DIR}/{DEMO_NAME}_jsd_by_comparison.pdf", dpi=300, bbox_inches="tight")
(/gpfs/data/steinruecken-lab/conda/lanaf/working_env) [t-9lanaf@cri22in001 my-singer-run]$ cat make_jsd_ooa.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tszip
from scipy.spatial.distance import jensenshannon

DATA_ROOT = "official_data2"
TRUTH_TREES_DIR = f"{DATA_ROOT}/diploid_sim/trees"
PLOTS_OUTPUT_DIR = f"{DATA_ROOT}/plots_combined/jsd"
RESULTS_DIR_WITH_POLEGON = "results"
RESULTS_DIR_WITHOUT_POLEGON = "results_np"

NUM_REPLICATES = 50
QUANTILE_PROBS = np.linspace(0, 1, 21)  # 21 edges -> 20 bins, ~5% truth mass each

DEMO_NAME = "out_of_africa"
TRUTH_NAMES = ["YRI", "CEU", "CHB"]
RECON_NAMES = ["DYRI", "DCEU", "DCHB"]
COMPARISON_INDEXES = [(0, 0), (0, 1), (1, 1), (1, 2), (0, 2), (2, 2)]
COMPARISON_LABELS = {
    (0, 0): "(YRI,YRI)",
    (0, 1): "(YRI,CEU)",
    (1, 1): "(CEU,CEU)",
    (1, 2): "(CEU,CHB)",
    (0, 2): "(YRI,CHB)",
    (2, 2): "(CHB,CHB)",
}


def results_dir(use_polegon):
    return RESULTS_DIR_WITH_POLEGON if use_polegon else RESULTS_DIR_WITHOUT_POLEGON


def get_named_sample_sets(ts, names):
    """Returns sample-node lists in `names` order, matched by population
    metadata name, or None if any name is missing from this tree."""
    pops = [None] * len(names)
    for pop in ts.populations():
        metadata = pop.metadata
        if metadata and "name" in metadata and metadata["name"] in names:
            pops[names.index(metadata["name"])] = pop
    if None in pops:
        return None
    return [list(ts.samples(population=pop.id)) for pop in pops]


def jsd(truth_props, recon_props):
    return jensenshannon(truth_props, recon_props, base=2) ** 2


# ---------------------------------------------------------------------------
# Ground truth, loaded once
# ---------------------------------------------------------------------------

truth_ts = tszip.decompress(f"{TRUTH_TREES_DIR}/{DEMO_NAME}.trees.tsz")
truth_sample_sets = get_named_sample_sets(truth_ts, TRUTH_NAMES)
if truth_sample_sets is None:
    raise RuntimeError(f"Could not find all of {TRUTH_NAMES} in the ground-truth tree.")

results = {}  # label -> (jsd_polegon, jsd_no_polegon)

for pop_comparison in COMPARISON_INDEXES:
    indexes = [pop_comparison]
    label = COMPARISON_LABELS[pop_comparison]

    edges = truth_ts.pair_coalescence_quantiles(
        quantiles=QUANTILE_PROBS, sample_sets=truth_sample_sets, indexes=indexes
    )[0]
    truth_counts = truth_ts.pair_coalescence_counts(
        time_windows=edges, sample_sets=truth_sample_sets, indexes=indexes, pair_normalise=False
    )[0]
    truth_props = truth_counts / truth_counts.sum()

    jsd_values = {}
    for use_polegon in (True, False):
        props_list = []
        n_skipped = 0
        for j in range(NUM_REPLICATES):
            tree_path = f"{DATA_ROOT}/{results_dir(use_polegon)}/{DEMO_NAME}/trees/{DEMO_NAME}.{j}.tsz"
            ts = tszip.decompress(tree_path)
            sample_sets = get_named_sample_sets(ts, RECON_NAMES)
            if sample_sets is None:
                n_skipped += 1
                continue
            counts = ts.pair_coalescence_counts(
                time_windows=edges, sample_sets=sample_sets, indexes=indexes, pair_normalise=False
            )[0]
            props_list.append(counts / counts.sum())
        if n_skipped:
            print(f"[{label}] {'Polegon' if use_polegon else 'no-Polegon'}: "
                  f"skipped {n_skipped}/{NUM_REPLICATES} replicates (missing population)")
        mean_props = np.mean(props_list, axis=0)
        jsd_values[use_polegon] = jsd(truth_props, mean_props)

    results[label] = (jsd_values[True], jsd_values[False])
    print(f"[{label}] JSD Polegon={results[label][0]:.4g}, JSD no-Polegon={results[label][1]:.4g}")


# ---------------------------------------------------------------------------
# Plot: one figure, comparisons listed along the y-axis
# ---------------------------------------------------------------------------

os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

labels = list(results.keys())
jsd_polegon = [results[l][0] for l in labels]
jsd_no_polegon = [results[l][1] for l in labels]

y = np.arange(len(labels))
height = 0.35

fig, ax = plt.subplots(figsize=(8, 0.6 * len(labels) + 2))
ax.barh(y - height / 2, jsd_polegon, height, label="SINGER + Polegon")
ax.barh(y + height / 2, jsd_no_polegon, height, label="SINGER")

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("Jensen-Shannon divergence")
ax.set_title(f"Pairwise coalescence JSD: Polegon vs. no-Polegon\nDemographic model: {DEMO_NAME}")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_OUTPUT_DIR}/{DEMO_NAME}_jsd_by_comparison.pdf", dpi=300, bbox_inches="tight")(/gpfs/data/steinruecken-lab/conda/lanaf/working_env)
