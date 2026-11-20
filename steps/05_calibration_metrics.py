"""Step 5: Calibration metrics (accuracy, Brier score, ECE) with 5 score bins."""

import argparse

import pandas as pd

from common import add_csv_args, default_labels_csv, load_graded

args = add_csv_args(argparse.ArgumentParser(description=__doc__)).parse_args()
graded = load_graded(args.scores_csv, args.labels_csv or default_labels_csv())

p = graded["p_fully_supported"]
y = graded["correct"]
graded["score_bin"] = pd.cut(p, bins=[0, .2, .4, .6, .8, 1], include_lowest=True)

bins = graded.groupby("score_bin", observed=True).agg(
    n=("correct", "size"),
    mean_score=("p_fully_supported", "mean"),
    observed_accuracy=("correct", "mean"),
)
ece = (
    bins["n"] * (bins["mean_score"] - bins["observed_accuracy"]).abs()
).sum() / len(graded)

print(f"Graded: {len(graded)} | Accuracy: {y.mean():.3f}")
print(f"Brier score: {((p - y) ** 2).mean():.3f} | ECE: {ece:.3f}")
print(bins.round(3).to_string())
