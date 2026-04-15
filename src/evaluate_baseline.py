"""
evaluate_baseline.py
Computes F1 and Exact Match on saved prediction files.
Breaks down by overall / answerable / unanswerable.
Compares against reported benchmarks to validate the pipeline.

Usage:
    python src/evaluate_baseline.py                  # evaluates full predictions
    python src/evaluate_baseline.py --sample         # evaluates 500-example sample
"""

import os
import json
import argparse
import string
import re
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
PREDICTIONS_DIR = os.path.join("outputs", "predictions")
RESULTS_DIR     = os.path.join("outputs", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Reported benchmarks for deepset/roberta-base-squad2 on SQuAD2.0 validation
# Source: https://huggingface.co/deepset/roberta-base-squad2
REPORTED_BENCHMARKS = {
    "squad": {
        "f1_overall" : 79.97,
        "em_overall" : 76.51,
    }
}


# ── SQuAD-style F1 / EM ───────────────────────────────────────────────────────

def normalize_answer(s):
    """Lowercase, strip punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(gold, pred):
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold, pred):
    gold_tokens = get_tokens(gold)
    pred_tokens = get_tokens(pred)
    common      = Counter(gold_tokens) & Counter(pred_tokens)
    num_same    = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def best_f1_em(gold_answers, pred):
    """Take the best F1/EM over all gold answers (SQuAD convention)."""
    if not gold_answers:
        # Unanswerable: correct iff model predicts empty
        correct = int(pred.strip() == "")
        return correct, correct
    f1 = max(compute_f1(g, pred) for g in gold_answers)
    em = max(compute_exact(g, pred) for g in gold_answers)
    return f1, em


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(predictions):
    """
    Compute F1 and EM overall and split by answerable / unanswerable.
    Returns a dict of metrics.
    """
    f1_all, em_all         = [], []
    f1_ans, em_ans         = [], []
    f1_unans, em_unans     = [], []

    for ex in predictions:
        pred         = ex["pred_answer"]
        gold_answers = ex["gold_answers"]
        is_answerable = ex["is_answerable"]

        f1, em = best_f1_em(gold_answers, pred)

        f1_all.append(f1)
        em_all.append(em)

        if is_answerable:
            f1_ans.append(f1)
            em_ans.append(em)
        else:
            f1_unans.append(f1)
            em_unans.append(em)

    def avg(lst):
        return round(100 * sum(lst) / len(lst), 2) if lst else 0.0

    return {
        "n_total"      : len(predictions),
        "n_answerable" : len(f1_ans),
        "n_unanswerable": len(f1_unans),
        "f1_overall"   : avg(f1_all),
        "em_overall"   : avg(em_all),
        "f1_answerable": avg(f1_ans),
        "em_answerable": avg(em_ans),
        "f1_unanswerable": avg(f1_unans),
        "em_unanswerable": avg(em_unans),
    }


def print_results(name, metrics, benchmark=None):
    print(f"\n  {'─'*44}")
    print(f"  Dataset : {name}")
    print(f"  {'─'*44}")
    print(f"  Total examples    : {metrics['n_total']}")
    print(f"  Answerable        : {metrics['n_answerable']}")
    print(f"  Unanswerable      : {metrics['n_unanswerable']}")
    print(f"\n  {'Metric':<25} {'Score':>8}")
    print(f"  {'─'*35}")
    print(f"  {'F1  (overall)':<25} {metrics['f1_overall']:>8.2f}")
    print(f"  {'EM  (overall)':<25} {metrics['em_overall']:>8.2f}")
    print(f"  {'F1  (answerable)':<25} {metrics['f1_answerable']:>8.2f}")
    print(f"  {'EM  (answerable)':<25} {metrics['em_answerable']:>8.2f}")
    print(f"  {'F1  (unanswerable)':<25} {metrics['f1_unanswerable']:>8.2f}")
    print(f"  {'EM  (unanswerable)':<25} {metrics['em_unanswerable']:>8.2f}")

    if benchmark:
        print(f"\n  {'─'*44}")
        print(f"  Pipeline check vs reported benchmarks:")
        for key, reported in benchmark.items():
            actual = metrics.get(key, 0)
            diff   = actual - reported
            status = "✓" if abs(diff) <= 1.5 else "✗ CHECK THIS"
            print(f"  {key:<25} yours={actual:.2f}  reported={reported:.2f}  diff={diff:+.2f}  {status}")


def print_section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(sample=False):
    suffix = "_sample" if sample else ""

    print_section("Baseline Evaluation")
    print(f"  Mode: {'sample (500 examples)' if sample else 'full dataset'}")

    all_results = {}

    for dataset, filename, benchmark in [
        ("SQuAD 2.0",  "squad_predictions.json", REPORTED_BENCHMARKS["squad"]),
        ("CUAD",       "cuad_predictions.json",  None),
    ]:
        path = os.path.join(PREDICTIONS_DIR, filename)

        if not os.path.exists(path):
            print(f"\n  [SKIP] {path} not found — run run_inference.py first")
            continue

        predictions = json.load(open(path))
        print_section(f"Evaluating {dataset}")
        print(f"  Loaded {len(predictions)} predictions from {path}")

        metrics = evaluate(predictions)
        print_results(dataset, metrics, benchmark=benchmark)
        all_results[dataset] = metrics

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, f"baseline_metrics{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print_section("Done")
    print(f"  Results saved → {out_path}")
    print(f"  Next step: Week 2 — subgroup analysis\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true",
                        help="Evaluate on 500-example sample predictions")
    args = parser.parse_args()
    main(sample=args.sample)