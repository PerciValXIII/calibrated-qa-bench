"""
run_all_models.py
Runs the full analysis pipeline across all three models:
  1. Subgroup tagging
  2. Calibration analysis (ECE per subgroup)
  3. Risk-coverage (AUC-RC per subgroup)
  4. Cross-model comparison tables and figures

Usage: python src/run_all_models.py
"""

import os
import json
import re
import string
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# ── Config ────────────────────────────────────────────────────────────────────
PREDICTIONS_DIR = os.path.join("outputs", "predictions")
FIGURES_DIR     = os.path.join("outputs", "figures", "cross_model")
RESULTS_DIR     = os.path.join("outputs", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = ["roberta-base", "roberta-large", "deberta"]
MODEL_LABELS = {
    "roberta-base" : "RoBERTa-base",
    "roberta-large": "RoBERTa-large",
    "deberta"      : "DeBERTa-v3",
}
DATASETS = ["squad", "cuad"]

N_BINS       = 10
N_THRESHOLDS = 100
MIN_COVERAGE = 0.05

SUBGROUP_AXES = [
    ("subgroup_qtype",         "Question Type"),
    ("subgroup_answer_length", "Answer Length"),
    ("subgroup_answerability", "Answerability"),
]

COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
          "#56B4E9", "#D55E00", "#F0E442", "#999999"]

MODEL_COLORS = {
    "roberta-base" : "#0072B2",
    "roberta-large": "#E69F00",
    "deberta"      : "#009E73",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def normalize_confidence(predictions):
    scores = np.array([ex["confidence"] for ex in predictions])
    return 1 / (1 + np.exp(-scores))


def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def f1_score(gold, pred):
    g = normalize_answer(gold).split()
    p = normalize_answer(pred).split()
    common = Counter(g) & Counter(p)
    if not common:
        return 0.0
    prec = sum(common.values()) / len(p)
    rec  = sum(common.values()) / len(g)
    return 2 * prec * rec / (prec + rec)


def is_correct(ex):
    pred  = ex["pred_answer"]
    golds = ex["gold_answers"]
    if not ex["is_answerable"]:
        return int(pred.strip() == "")
    if not golds:
        return 0
    return int(max(f1_score(g, pred) for g in golds) > 0.5)


# ── Subgroup Tagging ──────────────────────────────────────────────────────────

def get_question_type(question):
    q = question.strip().lower()
    if "highlight the parts" in q or "related to" in q:
        return "cuad_clause"
    first_word = q.split()[0] if q.split() else ""
    mapping = {
        "what": "What", "which": "What", "who": "Who", "whose": "Who",
        "when": "When", "where": "Where", "how": "How", "why": "Why",
        "is": "YesNo", "are": "YesNo", "was": "YesNo", "were": "YesNo",
        "did": "YesNo", "does": "YesNo", "do": "YesNo",
        "can": "YesNo", "could": "YesNo", "would": "YesNo",
    }
    return mapping.get(first_word, "Other")


def get_answer_length(gold_answers, pred_answer, is_answerable):
    if not is_answerable:
        return "none"
    text = gold_answers[0] if gold_answers else pred_answer
    if not text:
        return "none"
    n = len(text.split())
    if n <= 3:
        return "short"
    elif n <= 10:
        return "medium"
    return "long"


def tag_predictions(predictions, domain):
    tagged = []
    for ex in predictions:
        ex = dict(ex)
        ex["subgroup_domain"]        = domain
        ex["subgroup_qtype"]         = get_question_type(ex["question"])
        ex["subgroup_answer_length"] = get_answer_length(
            ex["gold_answers"], ex["pred_answer"], ex["is_answerable"]
        )
        ex["subgroup_answerability"] = "answerable" if ex["is_answerable"] else "unanswerable"
        tagged.append(ex)
    return tagged


# ── Calibration ───────────────────────────────────────────────────────────────

def compute_ece(confidences, correctness, n_bins=N_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc  = correctness[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(avg_conf - avg_acc)
    return float(ece)


# ── Risk-Coverage ─────────────────────────────────────────────────────────────

def compute_risk_coverage(confidences, correctness):
    thresholds = np.linspace(confidences.max(), confidences.min(), N_THRESHOLDS)
    coverages, risks = [], []
    for tau in thresholds:
        mask     = confidences >= tau
        coverage = mask.sum() / len(confidences)
        if coverage < MIN_COVERAGE:
            continue
        coverages.append(float(coverage))
        risks.append(float(1 - correctness[mask].mean()))
    return np.array(coverages), np.array(risks)


def compute_auc_rc(coverages, risks):
    if len(coverages) < 2:
        return float("nan")
    order = np.argsort(coverages)
    c, r  = coverages[order], risks[order]
    auc   = float(np.trapezoid(r, c))
    crange = c[-1] - c[0]
    return auc / crange if crange > 0 else float("nan")


# ── Per-model analysis ────────────────────────────────────────────────────────

def analyse_model_dataset(model_key, dataset_key):
    """Load predictions, tag, compute ECE + AUC-RC globally and per subgroup."""

    # Handle legacy filenames for roberta-base
    if model_key == "roberta-base":
        path = os.path.join(PREDICTIONS_DIR, f"{dataset_key}_predictions.json")
        if not os.path.exists(path):
            path = os.path.join(PREDICTIONS_DIR,
                                f"{model_key}_{dataset_key}_predictions.json")
    else:
        path = os.path.join(PREDICTIONS_DIR,
                            f"{model_key}_{dataset_key}_predictions.json")

    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None

    predictions = json.load(open(path))
    tagged      = tag_predictions(predictions, dataset_key)

    confidences = normalize_confidence(tagged)
    correctness = np.array([is_correct(ex) for ex in tagged])

    # Global
    global_ece    = compute_ece(confidences, correctness)
    covs, rsks    = compute_risk_coverage(confidences, correctness)
    global_auc_rc = compute_auc_rc(covs, rsks)

    results = {
        "__all__": {
            "ece"     : global_ece,
            "auc_rc"  : global_auc_rc,
            "accuracy": float(correctness.mean()),
            "n"       : len(tagged),
        }
    }

    # Per subgroup
    for axis_key, _ in SUBGROUP_AXES:
        groups = defaultdict(list)
        for i, ex in enumerate(tagged):
            groups[ex[axis_key]].append(i)

        results[axis_key] = {}
        for label, indices in sorted(groups.items()):
            if len(indices) < 20:
                continue
            c   = confidences[indices]
            a   = correctness[indices]
            ece = compute_ece(c, a)
            cv, rk = compute_risk_coverage(c, a)
            auc = compute_auc_rc(cv, rk)
            results[axis_key][label] = {
                "ece"     : ece,
                "auc_rc"  : auc,
                "accuracy": float(a.mean()),
                "n"       : len(indices),
            }

    return results


# ── Cross-model comparison plots ──────────────────────────────────────────────

def plot_cross_model_ece(all_results, dataset_key, axis_key, axis_name):
    """Grouped bar chart: ECE per subgroup, one bar per model."""
    # Collect all subgroup labels across models
    labels = set()
    for model_key in MODELS:
        if all_results[model_key][dataset_key] and axis_key in all_results[model_key][dataset_key]:
            labels.update(all_results[model_key][dataset_key][axis_key].keys())
    labels = sorted(labels)

    if not labels:
        return

    x     = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))

    for m_idx, model_key in enumerate(MODELS):
        eces = []
        for label in labels:
            val = (all_results[model_key][dataset_key]
                   .get(axis_key, {})
                   .get(label, {})
                   .get("ece", float("nan")))
            eces.append(val)

        bars = ax.bar(x + m_idx * width, eces, width,
                      label=MODEL_LABELS[model_key],
                      color=MODEL_COLORS[model_key], alpha=0.85)

    ax.set_xlabel("Subgroup", fontsize=11)
    ax.set_ylabel("ECE (lower = better)", fontsize=11)
    ax.set_title(
        f"{dataset_key.upper()} — ECE by {axis_name}\nCross-Model Comparison",
        fontsize=13, fontweight="bold"
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(
        FIGURES_DIR, f"{dataset_key}_ece_crossmodel_{axis_key}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_cross_model_global(all_results):
    """Bar chart: global ECE and AUC-RC for each model × dataset."""
    metrics   = ["ece", "auc_rc"]
    m_labels  = ["ECE (↓ better)", "AUC-RC (↓ better)"]
    datasets  = ["squad", "cuad"]
    d_labels  = ["SQuAD", "CUAD"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, m_label in zip(axes, metrics, m_labels):
        x     = np.arange(len(datasets))
        width = 0.25

        for m_idx, model_key in enumerate(MODELS):
            vals = []
            for ds in datasets:
                val = (all_results[model_key][ds]
                       .get("__all__", {})
                       .get(metric, float("nan")))
                vals.append(val)

            ax.bar(x + m_idx * width, vals, width,
                   label=MODEL_LABELS[model_key],
                   color=MODEL_COLORS[model_key], alpha=0.85)

        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel(m_label, fontsize=11)
        ax.set_title(f"Global {m_label}", fontsize=12, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(d_labels, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Cross-Model Calibration Comparison", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "global_crossmodel_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── Print comparison table ────────────────────────────────────────────────────

def print_comparison_table(all_results, dataset_key, axis_key, metric):
    """Print a clean terminal table: rows=subgroups, cols=models."""
    labels = set()
    for model_key in MODELS:
        if all_results[model_key][dataset_key]:
            labels.update(
                all_results[model_key][dataset_key].get(axis_key, {}).keys()
            )
    labels = sorted(labels)

    col_w = 14
    header = f"  {'Subgroup':<22}" + "".join(
        f"{MODEL_LABELS[m]:>{col_w}}" for m in MODELS
    )
    print(header)
    print(f"  {'─'*( 22 + col_w * len(MODELS))}")

    for label in labels:
        row = f"  {label:<22}"
        for model_key in MODELS:
            val = (all_results[model_key][dataset_key]
                   .get(axis_key, {})
                   .get(label, {})
                   .get(metric, float("nan")))
            row += f"{val:>{col_w}.4f}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print_section("Cross-Model Pipeline")

    # ── Run analysis for all model × dataset combinations ─────────────────────
    all_results = {m: {} for m in MODELS}

    for model_key in MODELS:
        print_section(f"Analysing {MODEL_LABELS[model_key]}")
        for dataset_key in DATASETS:
            print(f"\n  → {dataset_key.upper()}")
            result = analyse_model_dataset(model_key, dataset_key)
            all_results[model_key][dataset_key] = result or {}
            if result:
                print(f"    Global ECE    : {result['__all__']['ece']:.4f}")
                print(f"    Global AUC-RC : {result['__all__']['auc_rc']:.4f}")
                print(f"    Accuracy      : {result['__all__']['accuracy']:.4f}")
                print(f"    N examples    : {result['__all__']['n']}")

    # ── Cross-model comparison tables ─────────────────────────────────────────
    print_section("Cross-Model ECE Tables")

    for dataset_key in DATASETS:
        for axis_key, axis_name in SUBGROUP_AXES:
            print(f"\n  {dataset_key.upper()} — ECE by {axis_name}")
            print_comparison_table(all_results, dataset_key, axis_key, "ece")

    print_section("Cross-Model AUC-RC Tables")

    for dataset_key in DATASETS:
        for axis_key, axis_name in SUBGROUP_AXES:
            print(f"\n  {dataset_key.upper()} — AUC-RC by {axis_name}")
            print_comparison_table(all_results, dataset_key, axis_key, "auc_rc")

    # ── Global summary table ───────────────────────────────────────────────────
    print_section("Global Summary")
    print(f"\n  {'Model':<16} {'SQuAD ECE':>12} {'SQuAD AUC-RC':>14} "
          f"{'CUAD ECE':>12} {'CUAD AUC-RC':>14} {'SQuAD Acc':>12}")
    print(f"  {'─'*82}")
    for model_key in MODELS:
        sq = all_results[model_key].get("squad", {}).get("__all__", {})
        cu = all_results[model_key].get("cuad",  {}).get("__all__", {})
        print(f"  {MODEL_LABELS[model_key]:<16} "
              f"{sq.get('ece', float('nan')):>12.4f} "
              f"{sq.get('auc_rc', float('nan')):>14.4f} "
              f"{cu.get('ece', float('nan')):>12.4f} "
              f"{cu.get('auc_rc', float('nan')):>14.4f} "
              f"{sq.get('accuracy', float('nan')):>12.4f}")

    # ── Figures ────────────────────────────────────────────────────────────────
    print_section("Generating Cross-Model Figures")

    plot_cross_model_global(all_results)

    for dataset_key in DATASETS:
        for axis_key, axis_name in SUBGROUP_AXES:
            plot_cross_model_ece(all_results, dataset_key, axis_key, axis_name)

    # ── Save full results ──────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "cross_model_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print_section("Done")
    print(f"  Results saved → {out_path}")
    print(f"  Figures saved → {FIGURES_DIR}/\n")


if __name__ == "__main__":
    main()
