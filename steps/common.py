"""Shared configuration and helpers for the pipeline steps."""

import os
from getpass import getpass
from pathlib import Path

import pandas as pd


for _var in ("no_proxy", "NO_PROXY"):
    _hosts = [h for h in os.environ.get(_var, "").split(",") if h]
    if "googleapis.com" not in _hosts:
        os.environ[_var] = ",".join(_hosts + ["googleapis.com"])

# CSVs and figures are read from / written to the project root.
DATA_DIR = Path(__file__).resolve().parent.parent

DOMAIN = "Financial report"  # None = all domains; "Financial report" = finance only
GENERATOR = "gemini-3.1-flash-lite"  # Vertex AI (gcloud ADC)
SCORER = "gemini-decision-1"  # Gemini API only (API key); not on Vertex AI
JUDGE = "gemini-3.1-flash-lite"  # Vertex AI (gcloud ADC)

GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "XXXXX")
GCP_LOCATION = "global"

DATASET_BASE = "https://huggingface.co/datasets/MMDocIR/MMDocRAG/resolve/main"
API_URL = "https://generativelanguage.googleapis.com/v1beta"
SPLITS = {
    "dev": "dev_15.jsonl",
    "evaluation": "evaluation_15.jsonl",
}

scope = "all" if DOMAIN is None else DOMAIN.lower().replace(" ", "_").replace("/", "_")
prefix = f"mmdocrag_dev_eval_{scope}"
SCORES_CSV = DATA_DIR / f"{prefix}_scores.csv"
GRADING_CSV = DATA_DIR / f"{prefix}_grading.csv"
NLI_CSV = GRADING_CSV.with_name(f"{GRADING_CSV.stem}_nli.csv")


def vertex_client():
    """Gen AI SDK client on Vertex AI, using Application Default Credentials."""
    from google import genai

    return genai.Client(enterprise=True, project=GCP_PROJECT, location=GCP_LOCATION)


def get_api_key():
    """Gemini API key for gemini-decision-1: GEMINI_API_KEY, or prompt."""
    return os.environ.get("GEMINI_API_KEY") or getpass("Gemini API key: ").strip()


def save_csv(frame, filename):
    path = Path(filename)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def has_labels(path):
    if not Path(path).exists():
        return False
    correct = pd.read_csv(path, usecols=["correct"], dtype=str, keep_default_na=False)
    return correct["correct"].isin(["0", "1"]).any()


def default_labels_csv():
    """The most recently written labelled CSV: GRADING_CSV (step 3a) or NLI_CSV (step 3b)."""
    labelled = [path for path in (GRADING_CSV, NLI_CSV) if has_labels(path)]
    if not labelled:
        return GRADING_CSV
    return max(labelled, key=lambda path: path.stat().st_mtime)


def load_graded(scores_csv, labels_csv):
    """Merge support scores with correctness labels; keep graded rows only."""
    scores = pd.read_csv(scores_csv, dtype={"q_id": str})
    labels = pd.read_csv(labels_csv, dtype={"q_id": str})[["q_id", "correct"]]
    scored = scores.dropna(subset=["p_fully_supported"]).copy()
    graded = scored.merge(labels, on="q_id", how="left", validate="one_to_one")

    print(f"Scores: {scores_csv}\nLabels: {labels_csv}")
    print(f"Scored: {len(scored)} | Ungraded: {graded['correct'].isna().sum()}")
    graded = graded.dropna(subset=["correct"]).copy()
    if graded.empty:
        raise ValueError("No correctness labels found. Run step 3a or 3b first.")
    if not graded["correct"].isin([0, 1]).all():
        raise ValueError("The correct column must contain 0, 1, or blank.")

    graded["p_fully_supported"] = pd.to_numeric(
        graded["p_fully_supported"], errors="raise"
    )
    if not graded["p_fully_supported"].between(0, 1).all():
        raise ValueError("Support scores must be between 0 and 1.")
    graded["correct"] = graded["correct"].astype(int)
    return graded


def add_csv_args(parser):
    parser.add_argument("--scores-csv", type=Path, default=SCORES_CSV)
    parser.add_argument(
        "--labels-csv", type=Path, default=None,
        help="Defaults to the most recently written labelled grading CSV (step 3a or 3b).",
    )
    return parser
