"""Step 3b: Label answer correctness with bidirectional NLI entailment.

An answer counts as correct when the reference entails the candidate AND the
candidate entails the reference. Reads GRADING_CSV and writes NLI_CSV.
"""

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import GRADING_CSV, NLI_CSV

MODEL_ID = "cross-encoder/nli-deberta-v3-small"
BATCH_SIZE = 16
MAX_LENGTH = 512
ENTAILMENT = 1  # Model card: contradiction=0, entailment=1, neutral=2

grading = pd.read_csv(GRADING_CSV, dtype={"q_id": str}, keep_default_na=False)
if grading.empty:
    raise ValueError("The grading CSV is empty. Run step 2 first.")

# Start a fresh set of NLI labels for this run.
grading["correct"] = ""
grading["judge_model"] = ""
grading["judge_error"] = ""
grading["nli_reference_to_candidate"] = float("nan")
grading["nli_candidate_to_reference"] = float("nan")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():  # Apple Silicon GPU
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Loading {MODEL_ID} | device={device} | rows={len(grading)}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model = model.to(device).eval()

with torch.inference_mode():
    for start in tqdm(range(0, len(grading), BATCH_SIZE), desc="NLI grading"):
        batch = grading.iloc[start:start + BATCH_SIZE]
        references = (
            "Question: " + batch["question"]
            + "\nAnswer: " + batch["reference_answer"]
        ).tolist()
        candidates = (
            "Question: " + batch["question"]
            + "\nAnswer: " + batch["generated_answer"]
        ).tolist()

        premises = references + candidates
        hypotheses = candidates + references
        tokens = tokenizer(premises, hypotheses, truncation=False, verbose=False)
        size = len(batch)

        # Leave overlength rows ungraded to avoid dropping relevant details.
        valid = [
            i for i in range(size)
            if max(len(tokens["input_ids"][i]),
                   len(tokens["input_ids"][i + size])) <= MAX_LENGTH
        ]
        skipped = batch.index.difference(batch.index[valid])
        grading.loc[skipped, "judge_error"] = "NLI input exceeds 512 tokens"

        if valid:
            directions = valid + [i + size for i in valid]
            inputs = tokenizer.pad(
                {key: [values[i] for i in directions]
                 for key, values in tokens.items()},
                padding=True,
                return_tensors="pt",
            ).to(device)

            probabilities = model(**inputs).logits.softmax(dim=-1).cpu()
            forward, backward = probabilities.chunk(2)
            correct = (
                (forward.argmax(dim=-1) == ENTAILMENT)
                & (backward.argmax(dim=-1) == ENTAILMENT)
            )

            idx = batch.index[valid]
            grading.loc[idx, "correct"] = correct.int().numpy().astype(str)
            grading.loc[idx, "judge_model"] = MODEL_ID
            grading.loc[idx, "nli_reference_to_candidate"] = forward[:, ENTAILMENT].numpy()
            grading.loc[idx, "nli_candidate_to_reference"] = backward[:, ENTAILMENT].numpy()

grading.to_csv(NLI_CSV, index=False)

labeled = grading["correct"].isin(["0", "1"])
print(f"Saved {NLI_CSV}")
print(f"Labeled: {labeled.sum()}/{len(grading)}")
print(grading.loc[labeled, "correct"].value_counts().sort_index())
print(grading[[
    "q_id", "generated_answer", "reference_answer", "correct", "judge_error"
]].head(10).to_string())