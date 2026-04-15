"""
subgroup_tagger.py
Tags every prediction with subgroup labels across 4 axes:
  1. Question type  : What / Who / When / Where / How / Why / Other
  2. Answer length  : short (1-3 tokens) / medium (4-10) / long (10+) / none
  3. Answerability  : answerable / unanswerable
  4. Domain         : squad / cuad

Saves enriched prediction files to outputs/predictions/*_tagged.json

Usage: python src/subgroup_tagger.py
"""

import os
import json
import re
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
PREDICTIONS_DIR = os.path.join("outputs", "predictions")

INPUT_FILES = {
    "squad": "squad_predictions.json",
    "cuad" : "cuad_predictions.json",
}


# ── Subgroup Taggers ──────────────────────────────────────────────────────────

def get_question_type(question: str) -> str:
    """Extract question type from the first word of the question."""
    q = question.strip().lower()

    # CUAD questions follow a template — extract the category keyword instead
    if "highlight the parts" in q or "related to" in q:
        return "cuad_clause"

    first_word = q.split()[0] if q.split() else ""

    mapping = {
        "what"  : "What",
        "which" : "What",   # treat Which as What-type
        "who"   : "Who",
        "whose" : "Who",
        "when"  : "When",
        "where" : "Where",
        "how"   : "How",
        "why"   : "Why",
        "is"    : "YesNo",
        "are"   : "YesNo",
        "was"   : "YesNo",
        "were"  : "YesNo",
        "did"   : "YesNo",
        "does"  : "YesNo",
        "do"    : "YesNo",
        "can"   : "YesNo",
        "could" : "YesNo",
        "would" : "YesNo",
    }
    return mapping.get(first_word, "Other")


def get_answer_length(gold_answers: list, pred_answer: str, is_answerable: bool) -> str:
    """
    Bin answer length by token count.
    Uses gold answer if available, otherwise pred answer.
    """
    if not is_answerable:
        return "none"

    # Use first gold answer if available
    text = gold_answers[0] if gold_answers else pred_answer
    if not text:
        return "none"

    n_tokens = len(text.split())
    if n_tokens <= 3:
        return "short"
    elif n_tokens <= 10:
        return "medium"
    else:
        return "long"


def get_answerability(is_answerable: bool) -> str:
    return "answerable" if is_answerable else "unanswerable"


# ── Main Tagging ──────────────────────────────────────────────────────────────

def tag_predictions(predictions: list, domain: str) -> list:
    """Add subgroup fields to each prediction dict."""
    tagged = []
    for ex in tqdm(predictions, desc=f"  Tagging {domain}", leave=False):
        ex = dict(ex)  # copy

        ex["subgroup_domain"]        = domain
        ex["subgroup_qtype"]         = get_question_type(ex["question"])
        ex["subgroup_answer_length"] = get_answer_length(
            ex["gold_answers"], ex["pred_answer"], ex["is_answerable"]
        )
        ex["subgroup_answerability"] = get_answerability(ex["is_answerable"])

        tagged.append(ex)
    return tagged


def print_section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def print_distribution(tagged, domain):
    """Print subgroup distribution stats."""
    from collections import Counter

    print(f"\n  --- {domain} subgroup distribution ---")

    for axis, key in [
        ("Question type" , "subgroup_qtype"),
        ("Answer length" , "subgroup_answer_length"),
        ("Answerability" , "subgroup_answerability"),
    ]:
        counts = Counter(ex[key] for ex in tagged)
        total  = len(tagged)
        print(f"\n  {axis}:")
        for label, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {label:<20} {count:>6}  ({100*count/total:.1f}%)")


def main():
    print_section("Subgroup Tagger")

    for domain, filename in INPUT_FILES.items():
        path = os.path.join(PREDICTIONS_DIR, filename)

        if not os.path.exists(path):
            print(f"\n  [SKIP] {path} not found")
            continue

        print(f"\n  Loading {path}...")
        predictions = json.load(open(path))
        print(f"  Loaded {len(predictions)} predictions")

        tagged = tag_predictions(predictions, domain)
        print_distribution(tagged, domain)

        # Save enriched file
        out_path = os.path.join(PREDICTIONS_DIR, filename.replace(".json", "_tagged.json"))
        with open(out_path, "w") as f:
            json.dump(tagged, f)
        print(f"\n  Saved tagged predictions → {out_path}")

    print_section("Done")
    print("  Next step: python src/calibration_analysis.py\n")


if __name__ == "__main__":
    main()
