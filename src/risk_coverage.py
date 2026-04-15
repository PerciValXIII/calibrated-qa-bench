"""
risk_coverage.py
Computes selective prediction Risk-Coverage curves globally and per subgroup.
For each confidence threshold τ, computes:
  - Coverage : fraction of examples the model answers (confidence >= τ)
  - Risk     : error rate on the answered examples

Also computes AUC-RC (area under risk-coverage curve) per subgroup.

Usage: python src/risk_coverage.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import string
import re
from collections import defaultdict, Counter

# ── Config ────────────────────────────────────────────────────────────────────
PREDICTIONS_DIR = os.path.join("outputs", "predictions")
FIGURES_DIR     = os.path.join("outputs", "figures")
RESULTS_DIR     = os.path.join("outputs", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

N_THRESHOLDS = 100   # number of τ values to sweep
MIN_COVERAGE = 0.05  # ignore thresholds where coverage drops below this

SUBGROUP_AXES = [
    ("subgroup_qtype",         "Question Type"),
    ("subgroup_answer_length", "Answer Length"),
    ("subgroup_answerability", "Answerability"),
]

COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
          "#56B4E9", "#D55E00", "#F0E442", "#999999"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def normalize_confidence(predictions):
    scores = np.array([ex["confidence"] for ex in predictions])
    return 1 / (1 + np.exp(-scores))


def is_correct(ex):
    """F1 > 0.5 counts as correct. Empty pred on unanswerable = correct."""
    def normalize(s):
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = ''.join(c for c in s if c not in string.punctuation)
        return ' '.join(s.split())

    def f1(gold, pred):
        g = normalize(gold).split()
        p = normalize(pred).split()
        common = Counter(g) & Counter(p)
        if not common:
            return 0.0
        prec = sum(common.values()) / len(p)
        rec  = sum(common.values()) / len(g)
        return 2 * prec * rec / (prec + rec)

    pred  = ex["pred_answer"]
    golds = ex["gold_answers"]

    if not ex["is_answerable"]:
        return int(pred.strip() == "")
    if not golds:
        return 0
    return int(max(f1(g, pred) for g in golds) > 0.5)


def compute_risk_coverage(confidences, correctness, n_thresholds=N_THRESHOLDS):
    """
    Sweep confidence thresholds from max to min.
    At each threshold τ, answer only examples with confidence >= τ.
    Returns arrays of (coverage, risk, threshold).
    """
    thresholds = np.linspace(
        confidences.max(), confidences.min(), n_thresholds
    )

    coverages  = []
    risks      = []
    valid_taus = []

    for tau in thresholds:
        mask     = confidences >= tau
        coverage = mask.sum() / len(confidences)

        if coverage < MIN_COVERAGE:
            continue

        risk = 1 - correctness[mask].mean()

        coverages.append(float(coverage))
        risks.append(float(risk))
        valid_taus.append(float(tau))

    return (np.array(coverages), np.array(risks), np.array(valid_taus))


def compute_auc_rc(coverages, risks):
    """
    Area under the Risk-Coverage curve using trapezoidal integration.
    Lower is better — a perfect model has AUC-RC = 0.
    Normalise by coverage range so it's comparable across subgroups.
    """
    if len(coverages) < 2:
        return float("nan")
    # Sort by coverage ascending
    order = np.argsort(coverages)
    c = coverages[order]
    r = risks[order]
    auc = float(np.trapz(r, c))
    # Normalise by coverage range
    crange = c[-1] - c[0]
    return auc / crange if crange > 0 else float("nan")


def risk_at_coverage(coverages, risks, target_coverage):
    """Interpolate risk at a specific coverage level."""
    if len(coverages) == 0:
        return float("nan")
    idx = np.argmin(np.abs(coverages - target_coverage))
    return float(risks[idx])


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_global_rc(coverages, risks, auc, domain, save_path):
    """Single global risk-coverage curve."""
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(coverages, risks, color=COLORS[0], linewidth=2.5,
            label=f"AUC-RC = {auc:.4f}")
    ax.fill_between(coverages, risks, alpha=0.15, color=COLORS[0])

    # Reference: random baseline (constant risk = overall error rate)
    overall_risk = risks[np.argmax(coverages)]
    ax.axhline(overall_risk, color="gray", linestyle="--",
               linewidth=1.2, label=f"No abstention risk = {overall_risk:.3f}")

    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("Risk (Error Rate)", fontsize=12)
    ax.set_title(f"{domain} — Global Risk-Coverage Curve", fontsize=13,
                 fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(risks) * 1.1)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_subgroup_rc(subgroup_curves, axis_name, domain, save_path):
    """
    Overlay risk-coverage curves for all subgroups of one axis on a single plot.
    This is the key paper figure — shows subgroups hit same risk at different coverage.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for idx, (label, data) in enumerate(sorted(subgroup_curves.items())):
        c = data["coverages"]
        r = data["risks"]
        auc = data["auc_rc"]
        color = COLORS[idx % len(COLORS)]
        ax.plot(c, r, color=color, linewidth=2,
                label=f"{label} (AUC={auc:.3f})")
        ax.fill_between(c, r, alpha=0.05, color=color)

    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("Risk (Error Rate)", fontsize=12)
    ax.set_title(f"{domain} — Risk-Coverage by {axis_name}", fontsize=13,
                 fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_fixed_risk_coverage(subgroup_results_by_axis, domain, save_path):
    """
    Bar chart: at a fixed risk budget (5% and 10%), what coverage does each subgroup achieve?
    This is the most intuitive summary figure for the paper.
    """
    risk_budgets = [0.05, 0.10, 0.20]
    axes_to_plot = [
        ("subgroup_qtype",        "Question Type"),
        ("subgroup_answer_length","Answer Length"),
    ]

    fig, axes = plt.subplots(1, len(axes_to_plot),
                             figsize=(7 * len(axes_to_plot), 5))
    if len(axes_to_plot) == 1:
        axes = [axes]

    for ax_idx, (axis_key, axis_name) in enumerate(axes_to_plot):
        if axis_key not in subgroup_results_by_axis:
            continue

        subgroup_curves = subgroup_results_by_axis[axis_key]
        labels = sorted(subgroup_curves.keys())
        x      = np.arange(len(labels))
        width  = 0.25

        ax = axes[ax_idx]
        for b_idx, budget in enumerate(risk_budgets):
            coverages_at_budget = []
            for label in labels:
                data = subgroup_curves[label]
                c    = data["coverages"]
                r    = data["risks"]
                # Find max coverage where risk <= budget
                mask = r <= budget
                cov  = float(c[mask].max()) if mask.any() else 0.0
                coverages_at_budget.append(cov)

            bars = ax.bar(x + b_idx * width, coverages_at_budget,
                          width, label=f"Risk ≤ {int(budget*100)}%",
                          color=COLORS[b_idx], alpha=0.85)

        ax.set_xlabel("Subgroup", fontsize=11)
        ax.set_ylabel("Coverage Achievable", fontsize=11)
        ax.set_title(f"{domain} — Coverage at Fixed Risk\nby {axis_name}",
                     fontsize=12, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── Core Analysis ─────────────────────────────────────────────────────────────

def analyse_dataset(predictions, domain):
    print(f"\n  Computing risk-coverage for {len(predictions)} examples...")

    confidences = normalize_confidence(predictions)
    correctness = np.array([is_correct(ex) for ex in predictions])

    # ── Global curve ──────────────────────────────────────────────────────────
    coverages, risks, taus = compute_risk_coverage(confidences, correctness)
    auc = compute_auc_rc(coverages, risks)

    print(f"  Global AUC-RC       : {auc:.4f}")
    print(f"  Risk at 100% coverage: {risks[np.argmax(coverages)]:.4f}")
    print(f"  Risk at 80% coverage : {risk_at_coverage(coverages, risks, 0.8):.4f}")
    print(f"  Risk at 50% coverage : {risk_at_coverage(coverages, risks, 0.5):.4f}")

    plot_global_rc(
        coverages, risks, auc, domain,
        save_path=os.path.join(FIGURES_DIR, f"{domain.lower()}_rc_global.png")
    )

    results = {
        "__all__": {
            "auc_rc"  : auc,
            "risk_at_100": float(risks[np.argmax(coverages)]),
            "risk_at_80" : float(risk_at_coverage(coverages, risks, 0.8)),
            "risk_at_50" : float(risk_at_coverage(coverages, risks, 0.5)),
        }
    }

    # ── Subgroup curves ───────────────────────────────────────────────────────
    subgroup_results_by_axis = {}

    for axis_key, axis_name in SUBGROUP_AXES:
        print(f"\n  Subgroup: {axis_name}")

        groups = defaultdict(list)
        for i, ex in enumerate(predictions):
            groups[ex[axis_key]].append(i)

        subgroup_curves = {}
        print(f"  {'Subgroup':<22} {'n':>6}  {'AUC-RC':>8}  "
              f"{'Risk@80%':>10}  {'Risk@50%':>10}")
        print(f"  {'─'*60}")

        for label, indices in sorted(groups.items()):
            if len(indices) < 30:
                print(f"  {label:<22} skipped (n={len(indices)} < 30)")
                continue

            c_sub = confidences[indices]
            a_sub = correctness[indices]

            covs, rsks, _ = compute_risk_coverage(c_sub, a_sub)
            auc_sub       = compute_auc_rc(covs, rsks)
            r80           = risk_at_coverage(covs, rsks, 0.8)
            r50           = risk_at_coverage(covs, rsks, 0.5)

            subgroup_curves[label] = {
                "coverages": covs,
                "risks"    : rsks,
                "auc_rc"   : auc_sub,
                "n"        : len(indices),
            }

            print(f"  {label:<22} {len(indices):>6}  {auc_sub:>8.4f}  "
                  f"{r80:>10.4f}  {r50:>10.4f}")

        subgroup_results_by_axis[axis_key] = subgroup_curves
        results[axis_key] = {
            label: {
                "auc_rc"    : d["auc_rc"],
                "n"         : d["n"],
                "risk_at_80": float(risk_at_coverage(d["coverages"], d["risks"], 0.8)),
                "risk_at_50": float(risk_at_coverage(d["coverages"], d["risks"], 0.5)),
            }
            for label, d in subgroup_curves.items()
        }

        # Overlay plot
        if subgroup_curves:
            plot_subgroup_rc(
                subgroup_curves, axis_name, domain,
                save_path=os.path.join(
                    FIGURES_DIR,
                    f"{domain.lower()}_rc_{axis_key}.png"
                )
            )

    # Fixed-risk bar chart
    if subgroup_results_by_axis:
        plot_fixed_risk_coverage(
            subgroup_results_by_axis, domain,
            save_path=os.path.join(
                FIGURES_DIR, f"{domain.lower()}_rc_fixed_risk.png"
            )
        )

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print_section("Risk-Coverage Analysis")

    all_results = {}

    for domain, filename in [
        ("SQuAD", "squad_predictions_tagged.json"),
        ("CUAD",  "cuad_predictions_tagged.json"),
    ]:
        path = os.path.join(PREDICTIONS_DIR, filename)
        if not os.path.exists(path):
            print(f"\n  [SKIP] {path} not found — run subgroup_tagger.py first")
            continue

        print_section(f"Analysing {domain}")
        predictions  = json.load(open(path))
        results      = analyse_dataset(predictions, domain)
        all_results[domain] = results

    # Save metrics
    out_path = os.path.join(RESULTS_DIR, "risk_coverage_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print_section("Done")
    print(f"  Metrics saved → {out_path}")
    print(f"  Figures saved → {FIGURES_DIR}/")
    print(f"  Next step     : python src/run_inference.py --model roberta-large\n")


if __name__ == "__main__":
    main()
