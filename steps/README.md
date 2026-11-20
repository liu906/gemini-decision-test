# Pipeline steps (split from `test_decision.ipynb`)

Run from anywhere; CSVs and figures are read from / written to the project root.
Shared settings (domain, models, file names) live in `common.py`.
Gemini text generation (steps 2 and 3a) runs on Vertex AI via the Gen AI SDK with
`gcloud auth application-default login` credentials (project in `common.py`).
`gemini-decision-1` (steps 0 and 2) is only on the Gemini API: set `GEMINI_API_KEY`
or enter it at the prompt.

Setup: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt`

| Step | Script | Notebook cell |
|------|--------|---------------|
| 0 | `00_smoke_test_due_diligence.py` – hand-written due-diligence cases | 0 |
| 1 | `01_count_dataset.py` – count questions per split in `DOMAIN` | 2 |
| 2 | `02_generate_and_score.py` – generate answers + decision-1 support scores (resumable) | 3 |
| 3a | `03a_grade_llm_judge.py` – correctness labels via LLM judge (Vertex AI) | 4 (was commented out) |
| 3b | `03b_grade_nli.py` – correctness labels via bidirectional NLI | 5 |
| 4 | `04_evaluate_threshold.py` – confusion matrix, recall, FAR, F1, AUROC | 6 |
| 5 | `05_calibration_metrics.py` – Brier score, ECE | 7 |
| 6 | `06_calibration_plot.py` – reliability diagram | 8 |

Use either 3a or 3b. Steps 4–6 use whichever labelled grading CSV was written most recently. To use other files, pass `--scores-csv` / `--labels-csv`.
