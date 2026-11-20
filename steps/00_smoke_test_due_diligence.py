"""Step 0: Smoke-test gemini-decision-1 on hand-written due-diligence cases."""
import json

import pandas as pd
import requests

from common import DATA_DIR, get_api_key

API_URL = "https://generativelanguage.googleapis.com/v1beta/decisions"

OIL_GAS_POLICY = (
    "Companies that operate oil- or gas-producing wells are in scope."
)
OIL_GAS_QUESTION = "Is Orion in scope of the Oil & Gas industry policy?"
OIL_GAS_ANSWER = "Yes. Orion operates gas-producing wells."

COAL_POLICY = (
    "Companies with at least 10% of consolidated revenue "
    "from thermal coal extraction are in scope."
)
COAL_QUESTION = "Is Mira in scope of the Thermal Coal industry policy?"
COAL_DISCLOSURE = (
    "Mira earns 14% of consolidated revenue from coal extraction. "
    "The report gives no breakdown by coal type."
)

# Each row contains case name, question, generated answer, policy,
# company evidence, and expected support.
cases = [
    (
        "01_direct_support",
        OIL_GAS_QUESTION,
        OIL_GAS_ANSWER,
        OIL_GAS_POLICY,
        "Orion operates 12 producing natural gas wells.",
        "high",
    ),
    (
        "02_direct_contradiction",
        OIL_GAS_QUESTION,
        OIL_GAS_ANSWER,
        OIL_GAS_POLICY,
        "Orion's only business is selling pressure sensors to gas producers. "
        "It does not operate any wells.",
        "low",
    ),
    (
        "03_relevant_but_insufficient",
        OIL_GAS_QUESTION,
        OIL_GAS_ANSWER,
        OIL_GAS_POLICY,
        "Orion reports exposure to the natural gas sector. "
        "Its role and operating activities are not disclosed.",
        "low",
    ),
    (
        "04_supported_out_of_scope",
        "Is Northstar in scope of the Gambling industry policy?",
        "No. Northstar only sells software to betting operators, "
        "which the policy excludes.",
        "Betting operators are in scope. Software-only suppliers "
        "that do not accept wagers are excluded.",
        "Northstar's sole business is licensing odds software. "
        "It does not accept wagers.",
        "high",
    ),
    (
        "05_missing_revenue_breakdown",
        COAL_QUESTION,
        "Yes. Mira meets the 10% thermal coal revenue threshold.",
        COAL_POLICY,
        COAL_DISCLOSURE,
        "low",
    ),
    (
        "06_ownership_below_threshold",
        "Is Aster in scope of the Defence industry policy?",
        "Yes. The policy includes Aster because its subsidiary "
        "manufactures military radar.",
        "A parent company is in scope if it or a majority-owned "
        "subsidiary manufactures military radar.",
        "Aster owns 40% of a military radar manufacturer. "
        "The report identifies no other relevant operations.",
        "low",
    ),
    (
        "07_unsupported_extra_claim",
        "Is Lumen in scope of the Tobacco industry policy?",
        "Yes. Lumen is in scope because it manufactures cigarettes "
        "and earns 18% of its revenue from them.",
        "Cigarette manufacturers are in scope.",
        "Lumen manufactures cigarettes. "
        "The report does not disclose cigarette revenue.",
        "low",
    ),
    (
        "08_supported_insufficient_information",
        COAL_QUESTION,
        "The supplied information is insufficient to determine whether "
        "Mira is in scope because it does not disclose thermal coal "
        "revenue separately.",
        COAL_POLICY,
        COAL_DISCLOSURE,
        "high",
    ),
]

questions = {
    "fully_supported": {
        "type": "noul",
        "instructions": (
            "Using only the supplied citations, do they fully support "
            "every material claim in the generated answer to the question?"
        ),
    }
}

rows = []
with requests.Session() as session:
    session.headers["x-goog-api-key"] = get_api_key()

    for case_id, question, answer, policy, evidence, expected in cases:
        example = {
            "question": question,
            "generated_answer": answer,
            "citations": [f"Policy: {policy}", f"Company: {evidence}"],
        }
        row = {
            "case": case_id,
            "expected_support": expected,
            "p_fully_supported": None,
            "decision_confidence": None,
            "error": "",
            "question": question,
            "generated_answer": answer,
            "policy": policy,
            "company_evidence": evidence,
        }

        try:
            response = session.post(
                API_URL,
                json={
                    "model": "gemini-decision-1",
                    "input": json.dumps(example, ensure_ascii=False),
                    "questions": questions,
                },
                timeout=60,
            )
            if response.ok:
                decision = response.json()["answers"]["fully_supported"]
                row["p_fully_supported"] = decision["noul"]
                row["decision_confidence"] = decision["confidence"]
            else:
                row["error"] = f"HTTP {response.status_code}: {response.text}"
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            row["error"] = str(exc)

        rows.append(row)

results = pd.DataFrame(rows)
print(results[[
    "case",
    "expected_support",
    "p_fully_supported",
    "decision_confidence",
    "error",
]].to_string())

results.to_csv(DATA_DIR / "gemini_due_diligence_results.csv", index=False)

# Uncomment to download the CSV to your computer.
# from google.colab import files
# files.download("gemini_due_diligence_results.csv")