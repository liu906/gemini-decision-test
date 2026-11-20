"""Step 2: Generate answers on Vertex AI and score evidence support with gemini-decision-1.

Generation uses gcloud Application Default Credentials; gemini-decision-1 is only
available on the Gemini API, so scoring uses GEMINI_API_KEY (or a prompt).

Writes SCORES_CSV (checkpointed after every question; rerun to resume)
and GRADING_CSV (rows awaiting correctness labels from step 3a or 3b).
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from google.genai import errors, types

from common import (
    API_URL,
    DATASET_BASE,
    DOMAIN,
    GENERATOR,
    GRADING_CSV,
    SCORER,
    SCORES_CSV,
    SPLITS,
    get_api_key,
    save_csv,
    vertex_client,
)


def candidate_evidence(row):
    evidence = []
    for field, text_field, source in (
        ("text_quotes", "text", "text_quote"),
        ("img_quotes", "img_description", "image_description"),
    ):
        for quote in row.get(field) or []:
            text = (quote.get(text_field) or "").strip()
            if text:
                evidence.append({
                    "id": quote["quote_id"],
                    "source": source,
                    "text": text,
                })
    return evidence


items = {}
for split, filename in SPLITS.items():
    print(f"Loading {split}…", flush=True)
    domains = Counter()
    selected = 0
    eligible = 0

    with requests.get(
        f"{DATASET_BASE}/{filename}", stream=True, timeout=(10, 120)
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.strip():
                continue

            item = json.loads(line)
            domain = item["domain"]
            domains[domain] += 1
            if DOMAIN is not None and domain != DOMAIN:
                continue

            selected += 1
            evidence = candidate_evidence(item)
            reference = (item.get("answer_short") or "").strip()
            if not reference or not evidence:
                continue

            q_id = f"{split}:{item['q_id']}"
            if q_id in items:
                raise ValueError(f"Duplicate question ID: {q_id}")

            items[q_id] = {
                "q_id": q_id,
                "dataset_q_id": item["q_id"],
                "split": split,
                "domain": domain,
                "doc_name": item["doc_name"],
                "question": item["question"],
                "reference_answer": reference,
                "evidence": evidence,
            }
            eligible += 1

    print(
        f"{split} | total={sum(domains.values())} | selected={selected} | "
        f"eligible={eligible} | excluded={selected - eligible}",
        flush=True,
    )

if not items:
    raise ValueError("No matching questions have both a reference answer and usable evidence.")

saved = {}
if Path(SCORES_CSV).exists():
    previous = pd.read_csv(SCORES_CSV, keep_default_na=False, dtype={"q_id": str})
    saved = previous.set_index("q_id", drop=False).to_dict("index")
    if not previous["generator"].eq(GENERATOR).all() or not previous["scorer"].eq(SCORER).all():
        raise ValueError("The checkpoint uses different models. Change prefix to start a new run.")
    if saved.keys() - items.keys():
        raise ValueError("The checkpoint contains questions outside the current dataset selection.")

pending = [
    q_id for q_id in items
    if saved.get(q_id, {}).get("p_fully_supported", "") == ""
]
completed = len(items) - len(pending)
print(f"\nEligible: {len(items)} | Already scored: {completed} | Pending: {len(pending)}")

if pending:
    client = vertex_client()
    with requests.Session() as session:
        session.headers["x-goog-api-key"] = get_api_key()

        def post(path, payload):
            response = session.post(
                f"{API_URL}/{path}", json=payload, timeout=(10, 90)
            )
            response.raise_for_status()
            return response.json()

        for index, q_id in enumerate(pending, start=1):
            item = items[q_id]
            example = {
                "question": item["question"],
                "evidence": item["evidence"],
            }
            row = {key: value for key, value in item.items() if key != "evidence"}
            answer = saved.get(q_id, {}).get("generated_answer", "")
            row.update(
                generated_answer=answer,
                p_fully_supported="",
                decision_confidence="",
                generator=GENERATOR,
                scorer=SCORER,
                evidence_setting="text_quotes_plus_image_descriptions",
                error="",
            )

            stop = False
            stage = "generation"
            try:
                if not answer:
                    print(f"[{index}/{len(pending)}] {q_id} | generating", flush=True)
                    prompt = (
                        "Answer the question using only the supplied evidence. "
                        "Give a concise answer and preserve relevant units, periods, "
                        "and qualifiers. If the evidence is insufficient, say so.\n\n"
                        + json.dumps(example, ensure_ascii=False)
                    )
                    answer = (client.models.generate_content(
                        model=GENERATOR,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0),
                    ).text or "").strip()
                    if not answer:
                        raise ValueError("The generator returned an empty answer.")
                    row["generated_answer"] = answer

                stage = "scoring"
                print(f"[{index}/{len(pending)}] {q_id} | scoring", flush=True)
                decision = post(
                    "decisions",
                    {
                        "model": SCORER,
                        "input": json.dumps(
                            {**example, "generated_answer": answer},
                            ensure_ascii=False,
                        ),
                        "questions": {
                            "fully_supported": {
                                "type": "noul",
                                "instructions": (
                                    "Does the supplied evidence fully support the "
                                    "generated answer to the question? Check all material "
                                    "claims, including values, units, periods, and entities."
                                ),
                            }
                        },
                    },
                )
                verdict = decision["answers"]["fully_supported"]
                score = float(verdict["noul"])
                if not 0 <= score <= 1:
                    raise ValueError(f"Invalid support score: {score}")
                row["p_fully_supported"] = score
                row["decision_confidence"] = verdict.get("confidence")
                completed += 1

            except errors.APIError as exc:
                row["error"] = f"{stage}: HTTP {exc.code}: {exc.message}"
                stop = exc.code in {400, 401, 403, 404, 429}
            except requests.HTTPError as exc:
                status = exc.response.status_code
                row["error"] = f"{stage}: HTTP {status}: {exc.response.text[:500]}"
                stop = status in {401, 403, 404, 429}
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
                row["error"] = f"{stage}: {exc}"

            saved[q_id] = row
            save_csv(pd.DataFrame(saved.values()), SCORES_CSV)
            status = row["error"] or f"p_fully_supported={row['p_fully_supported']}"
            print(f"  Saved | {completed}/{len(items)} scored | {status}", flush=True)

            if stop:
                print("Stopped on API error. Rerun this script to resume.", flush=True)
                break

results = pd.DataFrame(saved.values())
has_score = pd.to_numeric(results["p_fully_supported"], errors="coerce").notna()
grading = results.loc[has_score, [
    "q_id", "dataset_q_id", "split", "domain",
    "question", "generated_answer", "reference_answer",
]].copy()

if Path(GRADING_CSV).exists():
    previous_grading = pd.read_csv(GRADING_CSV, dtype=str, keep_default_na=False)
    match_on = ["q_id", "question", "generated_answer", "reference_answer"]
    judge_columns = [
        column for column in ("correct", "judge_model", "judge_error")
        if column in previous_grading
    ]
    grading = grading.merge(
        previous_grading[match_on + judge_columns],
        on=match_on,
        how="left",
        validate="one_to_one",
    )

if "correct" not in grading:
    grading["correct"] = ""
else:
    grading["correct"] = grading["correct"].fillna("")

save_csv(grading, GRADING_CSV)
print(f"\nScored {len(grading)}/{len(items)} eligible questions.")
print(f"Saved {SCORES_CSV} and {GRADING_CSV}")
print(grading.groupby("split").size().rename("scored").to_frame().to_string())
print(grading.head(10).to_string())