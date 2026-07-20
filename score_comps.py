import json
import os
import sys
import time

import pandas as pd
from anthropic import Anthropic, APIError, APIStatusError

INPUT_CSV = "fintech.csv"
PROGRESS_FILE = "scores_progress.jsonl"
OUTPUT_CSV = "fintech_scored.csv"
MODEL = "claude-sonnet-4-6"

DIMENSIONS = {
    "management_team": {
        "weight": 0.30,
        "label": "Management Team",
        "rubric": (
            "Strength, completeness, and credibility of the founding/leadership team. "
            "Score higher for: repeat founders, notable prior employers or exits, relevant "
            "domain expertise, advisory board mentions, technical + commercial balance on the team, "
            "named credentials (PhDs, senior titles at recognizable companies). "
            "Score lower for: vague or missing team info, no named individuals, generic bios."
        ),
    },
    "market_opportunity": {
        "weight": 0.25,
        "label": "Size of Opportunity",
        "rubric": (
            "How large and how clearly defined the addressable market/problem is. "
            "Score higher for: specific market sizing (numbers, TAM/SAM language), a clearly "
            "articulated and significant problem, evidence of a growing or large market. "
            "Score lower for: vague claims of 'huge market', no quantification, narrow/niche "
            "framing with no growth story."
        ),
    },
    "product_technology": {
        "weight": 0.15,
        "label": "Product / Technology Strength",
        "rubric": (
            "Differentiation and defensibility of the product or technology. "
            "Score higher for: specific technical claims, proprietary technology, patents/IP "
            "mentions, clear product description, evidence of innovation. "
            "Score lower for: generic 'platform' language with no specifics, no differentiation "
            "from competitors mentioned, buzzword-heavy with no substance."
        ),
    },
    "competitive_environment": {
        "weight": 0.10,
        "label": "Competitive Environment",
        "rubric": (
            "Evidence of competitive positioning and barriers to entry. "
            "Score higher for: explicit comparison to or awareness of competitors, stated "
            "competitive advantage, barriers to entry (network effects, regulatory moat, data moat). "
            "Score lower for: no acknowledgment of competition, or competition mentioned with no "
            "stated differentiation."
        ),
    },
    "marketing_sales": {
        "weight": 0.10,
        "label": "Marketing / Sales Channels & Traction",
        "rubric": (
            "Evidence of go-to-market execution and real traction. "
            "Score higher for: named customers/partners, revenue or growth figures, awards, "
            "press coverage, stated distribution channels or partnerships. "
            "Score lower for: no traction evidence, purely aspirational language ('we aim to', "
            "'we will')."
        ),
    },
    "potential": {
        "weight": 0.10,
        "label": "Potential",
        "rubric": (
            "Signals about capital trajectory and plausible path to a profitable outcome "
            "(further funding rounds, acquisition, IPO). "
            "Score higher for: explicit mention of funding raised/sought, growth ambitions "
            "suggesting scalability, international expansion plans. "
            "Score lower for: no forward-looking statements, purely lifestyle/small-scale framing."
        ),
    },
}

SYSTEM_PROMPT = (
    """You are an experienced venture capital analyst applying the Payne Scorecard \
Method to evaluate early-stage companies based on their own self-published marketing text \
(an "overview", a "story", and "team" bios scraped from a startup directory website).

Important context and caveats you must keep in mind:
- This text is self-written promotional copy, not verified due diligence material. You are \
scoring the STRENGTH OF THE NARRATIVE AND DISCLOSED SIGNALS, not verified company performance.
- Absence of evidence should be scored low/neutral, not assumed positive. Do not give credit for \
things the text does not actually say.
- Be skeptical of generic buzzwords (e.g. "innovative", "cutting-edge", "world-class") that are \
not backed by specifics; these should not raise a score on their own.

For each of the following 6 dimensions, give a score from 0 to 100 (0 = no positive signal at all, \
50 = generic/unremarkable, 100 = exceptionally strong, specific signal) and a one-sentence justification \
referencing the specific content that drove the score (paraphrase, do not quote verbatim more than a \
few words).

Dimensions:
"""
    + "\n".join(
        f"- {d['label']} ({key}): {d['rubric']}" for key, d in DIMENSIONS.items()
    )
    + """

Respond ONLY with valid JSON, no markdown fences, no preamble, in this exact structure:
{
  "management_team": {"score": <int>, "justification": "<string>"},
  "market_opportunity": {"score": <int>, "justification": "<string>"},
  "product_technology": {"score": <int>, "justification": "<string>"},
  "competitive_environment": {"score": <int>, "justification": "<string>"},
  "marketing_sales": {"score": <int>, "justification": "<string>"},
  "exit_potential": {"score": <int>, "justification": "<string>"}
}
"""
)


def build_user_prompt(row) -> str:
    return f"""Company: {row["company_name"]}
Sector: {row["sector"]}
Trading for: {row["trading_for"]}
Employees: {row["employees"]}
Funding stage (context only — do not let this bias your reading of the TEXT itself): {row["funding_stage"]}

--- OVERVIEW ---
{row["overview"]}

--- STORY ---
{row["story"]}

--- TEAM ---
{row["team"]}
"""


def parse_response(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned)


def score_company(client: Anthropic, row, max_retries: int = 5) -> dict:
    prompt = build_user_prompt(row)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            return parse_response(text)
        except APIStatusError as e:
            last_err = e
            wait = min(2**attempt, 30)
            if e.status_code == 429:
                print(f"    rate limited, waiting {wait}s...")
            else:
                print(
                    f"    API error {e.status_code} (attempt {attempt + 1}/{max_retries}): {e}"
                )
            time.sleep(wait)
        except (APIError, json.JSONDecodeError) as e:
            last_err = e
            print(f"    error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Failed after {max_retries} attempts: {last_err}")


def weighted_total(scores: dict) -> float:
    return round(
        sum(scores[k]["score"] * DIMENSIONS[k]["weight"] for k in DIMENSIONS), 1
    )


def load_progress() -> dict:
    done = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["company_name"]] = rec
    return done


def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("Set ANTHROPIC_API_KEY environment variable first.")

    df = pd.read_csv(INPUT_CSV, index_col=0).fillna("Not stated")
    client = Anthropic()

    done = load_progress()
    print(
        f"Resuming: {len(done)}/{len(df)} already scored."
        if done
        else f"Starting fresh: {len(df)} companies."
    )

    with open(PROGRESS_FILE, "a") as progress_f:
        for i, row in df.iterrows():
            name = row["company_name"]
            if name in done:
                continue
            print(f"[{len(done) + 1}/{len(df)}] Scoring {name}...")
            try:
                scores = score_company(client, row)
            except RuntimeError as e:
                print(f"  GAVE UP on {name}: {e}")
                continue
            record = {"company_name": name, "scores": scores}
            progress_f.write(json.dumps(record) + "\n")
            progress_f.flush()
            done[name] = record
            print(f"  weighted total: {weighted_total(scores)}")

    # Build final output CSV
    rows = []
    for i, row in df.iterrows():
        rec = done.get(row["company_name"])
        out = row.to_dict()
        if rec:
            scores = rec["scores"]
            for dim in DIMENSIONS:
                out[f"score_{dim}"] = scores[dim]["score"]
                out[f"justification_{dim}"] = scores[dim]["justification"]
            out["payne_weighted_score"] = weighted_total(scores)
        else:
            for dim in DIMENSIONS:
                out[f"score_{dim}"] = None
                out[f"justification_{dim}"] = None
            out["payne_weighted_score"] = None
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Wrote {OUTPUT_CSV} ({len(out_df)} rows).")
    scored = out_df["payne_weighted_score"].notna().sum()
    print(f"Successfully scored: {scored}/{len(out_df)}")


if __name__ == "__main__":
    main()
