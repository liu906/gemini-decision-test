"""Step 0b: Smoke-test gemini-decision-1 on image evidence (tables and charts).

Each case pairs a question and generated answer with evidence in which the
deciding facts appear only in a synthetic image, so the expected support is
known. Controls check that the model reads the image rather than trusting the
answer: the same facts as text (should match) and the image removed (should drop).

Usage:
  python steps/00b_smoke_test_multimodal.py            # needs GEMINI_API_KEY (or prompt)
  python steps/00b_smoke_test_multimodal.py --dry-run  # only render images and requests
"""

import argparse
import base64
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import DATA_DIR, SCORER, get_api_key

IMAGE_DIR = DATA_DIR / "smoke_test_images"
RESULTS_CSV = DATA_DIR / "gemini_multimodal_smoke_results.csv"

COAL_POLICY = (
    "Policy: Companies with at least 10% of consolidated revenue "
    "from thermal coal extraction are in scope."
)
COAL_QUESTION = "Is Mira in scope of the Thermal Coal industry policy?"

INSTRUCTIONS = (
    "Using only the supplied citations, including any images, do they fully "
    "support every material claim in the generated answer to the question?"
)


# ---------------------------------------------------------------- images

def save_table(name, title, header, rows):
    fig, ax = plt.subplots(figsize=(6, 0.6 + 0.45 * (len(rows) + 1)), dpi=150)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width(list(range(len(header))))  # no clipped labels
    table.scale(1, 1.6)
    fig.savefig(IMAGE_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def save_bars(name, title, labels, values, ylabel):
    fig, ax = plt.subplots(figsize=(5, 3.4), dpi=150)
    bars = ax.bar(labels, values, color="#4c72b0")
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=3)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(IMAGE_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def render_images():
    IMAGE_DIR.mkdir(exist_ok=True)
    header = ["Segment", "FY2024 revenue (USD m)", "Share of total"]
    save_table("mira_revenue_14pct", "Mira Corp. - Revenue by segment, FY2024", header, [
        ["Thermal coal extraction", "140", "14%"],
        ["Metallurgical coal", "260", "26%"],
        ["Logistics services", "600", "60%"],
        ["Total", "1,000", "100%"],
    ])
    save_table("mira_revenue_6pct", "Mira Corp. - Revenue by segment, FY2024", header, [
        ["Thermal coal extraction", "60", "6%"],
        ["Metallurgical coal", "340", "34%"],
        ["Logistics services", "600", "60%"],
        ["Total", "1,000", "100%"],
    ])
    # Shares are deliberately omitted: the 15% must be computed (300 / 2,000).
    save_table("mira_revenue_no_shares", "Mira Corp. - Revenue by segment, FY2024",
               ["Segment", "FY2024 revenue (USD m)"], [
        ["Thermal coal extraction", "300"],
        ["Metallurgical coal", "500"],
        ["Logistics services", "1,200"],
        ["Total", "2,000"],
    ])
    save_bars("orion_revenue_bars", "Orion Ltd. - Annual revenue",
              ["2021", "2022", "2023"], [120, 135, 150], "Revenue (USD m)")
    save_bars("orion_headcount_bars", "Orion Ltd. - Employees",
              ["2021", "2022", "2023"], [410, 395, 430], "Headcount")


# ----------------------------------------------------------------- cases

REVENUE_Q = "How did Orion's annual revenue change from 2021 to 2023?"

# (case, question, answer, text citations, image or None, expected support)
CASES = [
    ("01_table_supports", COAL_QUESTION,
     "Yes. Thermal coal extraction is 14% of Mira's revenue, above the 10% threshold.",
     [COAL_POLICY], "mira_revenue_14pct", "high"),
    ("02_table_contradicts", COAL_QUESTION,
     "Yes. Thermal coal extraction is 14% of Mira's revenue, above the 10% threshold.",
     [COAL_POLICY], "mira_revenue_6pct", "low"),
    ("03_table_supports_no", COAL_QUESTION,
     "No. Thermal coal extraction is only 6% of Mira's revenue, below the 10% threshold.",
     [COAL_POLICY], "mira_revenue_6pct", "high"),
    ("04_table_calculation_correct", COAL_QUESTION,
     "Yes. Thermal coal extraction is USD 300m of USD 2,000m revenue (15%), "
     "above the 10% threshold.",
     [COAL_POLICY], "mira_revenue_no_shares", "high"),
    ("05_table_calculation_wrong", COAL_QUESTION,
     "Yes. Thermal coal extraction is USD 300m of USD 2,000m revenue (25%), "
     "above the 10% threshold.",
     [COAL_POLICY], "mira_revenue_no_shares", "low"),
    ("06_chart_supports", REVENUE_Q,
     "Revenue grew from USD 120m in 2021 to USD 150m in 2023.",
     [], "orion_revenue_bars", "high"),
    ("07_chart_wrong_direction", REVENUE_Q,
     "Revenue fell from USD 150m in 2021 to USD 120m in 2023.",
     [], "orion_revenue_bars", "low"),
    ("08_chart_wrong_value", REVENUE_Q,
     "Revenue grew from USD 120m in 2021 to USD 180m in 2023.",
     [], "orion_revenue_bars", "low"),
    ("09_irrelevant_image", REVENUE_Q,
     "Revenue grew from USD 120m in 2021 to USD 150m in 2023.",
     [], "orion_headcount_bars", "low"),
    # Controls for case 01.
    ("10_control_same_facts_as_text", COAL_QUESTION,
     "Yes. Thermal coal extraction is 14% of Mira's revenue, above the 10% threshold.",
     [COAL_POLICY, "Company: Mira's FY2024 revenue was USD 1,000m, of which thermal "
      "coal extraction was USD 140m (14%)."], None, "high"),
    ("11_control_image_removed", COAL_QUESTION,
     "Yes. Thermal coal extraction is 14% of Mira's revenue, above the 10% threshold.",
     [COAL_POLICY], None, "low"),
]


def build_input(question, answer, citations, image):
    parts = [{
        "type": "text",
        "text": json.dumps({
            "question": question,
            "generated_answer": answer,
            "citations": citations,
        }, ensure_ascii=False),
    }]
    if image:
        parts.append({"type": "text", "text": "Citation image:"})
        parts.append({
            "type": "image",
            "mime_type": "image/png",
            "data": base64.b64encode((IMAGE_DIR / f"{image}.png").read_bytes()).decode(),
        })
    return parts


# ------------------------------------------------------------------- run

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dry-run", action="store_true", help="Render images and requests; no API calls.")
args = parser.parse_args()

render_images()
print(f"Rendered images in {IMAGE_DIR}")

client = None
if not args.dry_run:
    from google import genai

    client = genai.Client(api_key=get_api_key())

rows = []
for case_id, question, answer, citations, image, expected in CASES:
    parts = build_input(question, answer, citations, image)
    row = {
        "case": case_id,
        "evidence": "image" if image else "text only",
        "expected_support": expected,
        "p_fully_supported": None,
        "decision_confidence": None,
        "error": "",
        "image": image or "",
        "question": question,
        "generated_answer": answer,
    }
    if args.dry_run:
        kinds = [p["type"] for p in parts]
        print(f"{case_id}: input parts={kinds}, bytes={len(json.dumps(parts)):,}")
    else:
        try:
            response = client.decisions.create(
                model=SCORER,
                input=parts,
                questions={"fully_supported": {"type": "noul", "instructions": INSTRUCTIONS}},
            )
            decision = response.answers["fully_supported"]
            row["p_fully_supported"] = decision.noul
            row["decision_confidence"] = decision.confidence
        except Exception as exc:  # report every failure in the table, keep going
            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        print(f"{case_id}: {row['error'] or row['p_fully_supported']}", flush=True)
    rows.append(row)

if not args.dry_run:
    results = pd.DataFrame(rows)
    p = pd.to_numeric(results["p_fully_supported"])
    results["pass"] = ((p >= 0.5) == results["expected_support"].eq("high")).where(p.notna())
    results.to_csv(RESULTS_CSV, index=False)

    print()
    print(results[[
        "case", "evidence", "expected_support", "p_fully_supported", "pass", "error",
    ]].to_string(index=False))
    scored = results["pass"].notna()
    print(f"\nPassed {int(results.loc[scored, 'pass'].sum())}/{scored.sum()} "
          f"(threshold 0.5) | errors: {(~scored).sum()}")
    print(f"Saved {RESULTS_CSV}")
