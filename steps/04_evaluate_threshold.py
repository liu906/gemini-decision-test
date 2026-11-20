"""Step 4: Confusion matrix and threshold metrics for gemini-decision-1 support scores."""

import argparse

import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from common import add_csv_args, default_labels_csv, load_graded

parser = add_csv_args(argparse.ArgumentParser(description=__doc__))
parser.add_argument("--threshold", type=float, default=0.5)
args = parser.parse_args()
THRESHOLD = args.threshold

data = load_graded(args.scores_csv, args.labels_csv or default_labels_csv())
p = data["p_fully_supported"]
y_true = data["correct"]
y_pred = (p >= THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

matrix = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=["Judge: Incorrect", "Judge: Correct"],
    columns=["Decision 1: Incorrect", "Decision 1: Correct"],
)
print(matrix.to_string())


# Return NaN when the relevant class is absent.
def ratio(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


metrics = pd.Series({
    "Accuracy": (tp + tn) / len(data),
    "Recall": ratio(tp, tp + fn),
    "FAR (false acceptance rate)": ratio(fp, fp + tn),
    "F1": ratio(2 * tp, 2 * tp + fp + fn),
    "AUROC": (
        roc_auc_score(y_true, p) if y_true.nunique() == 2 else float("nan")
    ),
}, name="Value")

print(f"\nThreshold: {THRESHOLD:.2f} | Evaluated: {len(data)}")
print(f"Correct answer rejected (FN): {fn}")
print(f"Incorrect answer accepted (FP): {fp}")
print(metrics.round(3).to_frame().to_string())
