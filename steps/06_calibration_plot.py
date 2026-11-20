"""Step 6: Reliability diagram with Wilson 95% intervals and per-bin counts."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA_DIR, add_csv_args, default_labels_csv, load_graded

parser = add_csv_args(argparse.ArgumentParser(description=__doc__))
parser.add_argument("--bins", type=int, default=10, help="Use 5 to match step 5's ECE bins.")
parser.add_argument("--output", type=Path, default=DATA_DIR / "mmdocrag_finance_calibration.png")
parser.add_argument("--show", action="store_true", help="Open the plot window.")
args = parser.parse_args()

graded = load_graded(args.scores_csv, args.labels_csv or default_labels_csv())

N_BINS = args.bins
edges = np.linspace(0, 1, N_BINS + 1)
centers = (edges[:-1] + edges[1:]) / 2
width = edges[1] - edges[0]

data = graded[["p_fully_supported", "correct"]].copy()
data["bin"] = pd.cut(
    data["p_fully_supported"], edges, labels=False, include_lowest=True
)
summary = data.groupby("bin").agg(
    n=("correct", "size"),
    mean_score=("p_fully_supported", "mean"),
    accuracy=("correct", "mean"),
)

# Wilson 95% confidence intervals for accuracy in nonempty bins.
n = summary["n"].to_numpy()
accuracy = summary["accuracy"].to_numpy()
z = 1.96
denominator = 1 + z**2 / n
midpoint = (accuracy + z**2 / (2 * n)) / denominator
half_width = z * np.sqrt(
    accuracy * (1 - accuracy) / n + z**2 / (4 * n**2)
) / denominator
lower = np.minimum(accuracy, midpoint - half_width)
upper = np.maximum(accuracy, midpoint + half_width)

fig, ax = plt.subplots(figsize=(5.3, 4.8), layout="constrained")
ax_count = ax.twinx()

# Put counts behind the calibration points.
ax_count.set_zorder(0)
ax.set_zorder(1)
ax.patch.set_visible(False)

counts = summary["n"].reindex(range(N_BINS), fill_value=0).to_numpy()
bars = ax_count.bar(
    centers, counts, width=0.85 * width, color="#dfdfdf", linewidth=0
)
ax_count.set_ylim(0, counts.max() * 1.18)
ax_count.set_ylabel("Questions", color="#666666")
ax_count.tick_params(axis="y", colors="#666666")
ax_count.yaxis.get_major_locator().set_params(integer=True)
ax_count.bar_label(
    bars, labels=[str(c) if c else "" for c in counts],
    padding=3, fontsize=8, color="#666666",
)

ax.plot([0, 1], [0, 1], "--", color="#666666", linewidth=1.4)
ax.errorbar(
    summary["mean_score"], accuracy,
    yerr=[accuracy - lower, upper - accuracy],
    fmt="o", color="#c51b32", markersize=6,
    elinewidth=1.5, capsize=3, zorder=3,
)

ax.set(
    title="Gemini Decision 1 · MMDocRAG Finance",
    xlabel="Predicted evidence support",
    ylabel="Observed accuracy (LLM judge)",
    xlim=(-0.025, 1.025),
    ylim=(-0.025, 1.025),
    xticks=[0, 0.25, 0.5, 0.75, 1],
    yticks=np.linspace(0, 1, 6),
)
ax.grid(color="#b0b0b0", alpha=0.2, linewidth=0.8)
ax.set_axisbelow(True)

fig.savefig(args.output, dpi=300, bbox_inches="tight")
print(f"Saved {args.output}")
if args.show:
    plt.show()