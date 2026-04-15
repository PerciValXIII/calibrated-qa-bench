"""
calibration_analysis.py
Computes calibration metrics and produces figures:
  - ECE (Expected Calibration Error) globally and per subgroup
  - Reliability diagrams (global + per subgroup)
  - Confidence distributions per subgroup

Usage: python src/calibration_analysis.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
PREDICTIONS_DIR = os.path.join("outputs", "predictions")
FIGURES_DIR     = os.path.join("outputs", "figures")
RESULTS_DIR     = os.path.join("outputs", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

N_BINS = 10  # calibration bins

SUBGROUP_AXES = [
    ("subgroup_qtype",          "Question Type"),
    ("subgroup_answer_length",  "Answer Length"),
    ("subgroup_answerability",  "Answerability"),
]

# Colourblind-friendly palette
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def normalize_confidence(predictions):
    """
    Normalize raw span scores to [0,1] using sigmoid.
    Raw logit sums can be any real number — sigmoid maps them to a probability.
    """
    scores = np.array([ex["confidence"] for ex in predictions])
    return 1 / (1 + np.exp(-scores))


def is_correct(ex):
    """
    Binary correctness: 1 if prediction matches any gold answer (F1 > 0.5),
    0 otherwise. For unanswerable, correct iff pred is empty.
    """
    import string, re
    from collections import Counter

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

    pred = ex["pred_answer"]
    golds = ex["gold_answers"]

    if not ex["is_answerable"]:
        return int(pred.strip() == "")

    if not golds:
        return 0

    return int(max(f1(g, pred) for g in golds) > 0.5)


def compute_ece(confidences, correctness, n_bins=N_BINS):
    """
    Expected Calibration Error.
    Bins predictions by confidence, measures |avg_confidence - avg_accuracy| per bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)

        if mask.sum() == 0:
            bin_data.append(None)
            continue

        avg_conf = confidences[mask].mean()
        avg_acc  = correctness[mask].mean()
        weight   = mask.sum() / len(confidences)
        ece     += weight * abs(avg_conf - avg_acc)

        bin_data.append({
            "bin_mid"  : (lo + hi) / 2,
            "avg_conf" : float(avg_conf),
            "avg_acc"  : float(avg_acc),
            "count"    : int(mask.sum()),
            "weight"   : float(weight),
        })

    return float(ece), bin_data


def compute_mce(confidences, correctness, n_bins=N_BINS):
    """Maximum Calibration Error — worst-case bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    mce  = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc  = correctness[mask].mean()
        mce = max(mce, abs(avg_conf - avg_acc))
    return float(mce)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_reliability_diagram(bin_data, ece, title, save_path, color=COLORS[0]):
    """Single reliability diagram."""
    valid_bins = [b for b in bin_data if b is not None]
    if not valid_bins:
        return

    bin_mids  = np.array([b["bin_mid"]  for b in valid_bins])
    avg_accs  = np.array([b["avg_acc"]  for b in valid_bins])
    avg_confs = np.array([b["avg_conf"] for b in valid_bins])
    bar_width = 0.9 / N_BINS

    fig, ax = plt.subplots(figsize=(5, 5))

    # Draw accuracy bars
    ax.bar(bin_mids, avg_accs, width=bar_width,
           color=color, alpha=0.85, label="Actual accuracy",
           zorder=2, align="center", edgecolor="white", linewidth=0.5)

    # Draw confidence gap on top (overconfidence = positive gap)
    gap = avg_confs - avg_accs
    ax.bar(bin_mids, np.maximum(gap, 0), bottom=avg_accs,
           width=bar_width, color="#D55E00", alpha=0.4,
           label="Overconfidence gap", zorder=3,
           align="center", edgecolor="white", linewidth=0.5)
    # Underconfidence gap (negative gap — accuracy exceeds confidence)
    ax.bar(bin_mids, np.maximum(-gap, 0), bottom=avg_confs,
           width=bar_width, color="#009E73", alpha=0.4,
           label="Underconfidence gap", zorder=3,
           align="center", edgecolor="white", linewidth=0.5)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2,
            label="Perfect calibration", zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_subgroup_reliability(subgroup_results, axis_name, domain, save_path):
    """Grid of reliability diagrams for all subgroups of one axis."""
    labels = [k for k in subgroup_results if k != "__all__"]
    n      = len(labels)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for idx, label in enumerate(labels):
        data  = subgroup_results[label]
        ax    = axes[idx]
        color = COLORS[idx % len(COLORS)]

        valid_bins = [b for b in data["bin_data"] if b is not None]
        if not valid_bins:
            ax.set_visible(False)
            continue

        bin_mids  = np.array([b["bin_mid"]  for b in valid_bins])
        avg_accs  = np.array([b["avg_acc"]  for b in valid_bins])
        avg_confs = np.array([b["avg_conf"] for b in valid_bins])
        bar_width = 0.9 / N_BINS
        gap       = avg_confs - avg_accs

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, zorder=1)
        ax.bar(bin_mids, avg_accs, width=bar_width, alpha=0.85,
               color=color, zorder=2, align="center",
               edgecolor="white", linewidth=0.5)
        ax.bar(bin_mids, np.maximum(gap, 0), bottom=avg_accs,
               width=bar_width, alpha=0.4, color="#D55E00",
               zorder=3, align="center", edgecolor="white", linewidth=0.5)
        ax.bar(bin_mids, np.maximum(-gap, 0), bottom=avg_confs,
               width=bar_width, alpha=0.4, color="#009E73",
               zorder=3, align="center", edgecolor="white", linewidth=0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}\nECE={data['ece']:.4f}  n={data['n']}", fontsize=11)
        ax.set_xlabel("Confidence", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.grid(alpha=0.3)

    # Hide unused axes
    for idx in range(len(labels), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{domain} — Reliability by {axis_name}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_confidence_distribution(predictions, confidences, domain, save_path):
    """Confidence score distributions for answerable vs unanswerable."""
    ans_conf   = confidences[[i for i, ex in enumerate(predictions) if ex["is_answerable"]]]
    unans_conf = confidences[[i for i, ex in enumerate(predictions) if not ex["is_answerable"]]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 30)

    ax.hist(ans_conf,   bins=bins, alpha=0.65, color=COLORS[0], label="Answerable",   density=True)
    ax.hist(unans_conf, bins=bins, alpha=0.65, color=COLORS[1], label="Unanswerable", density=True)

    ax.set_xlabel("Confidence (sigmoid-normalized)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{domain} — Confidence Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── Core Analysis ─────────────────────────────────────────────────────────────

def analyse_dataset(predictions, domain):
    print(f"\n  Computing correctness and confidence for {len(predictions)} examples...")

    confidences = normalize_confidence(predictions)
    correctness = np.array([is_correct(ex) for ex in predictions])

    print(f"  Overall accuracy  : {correctness.mean():.4f}")
    print(f"  Mean confidence   : {confidences.mean():.4f}")

    results = {}

    # ── Global ECE ────────────────────────────────────────────────────────────
    ece, bin_data = compute_ece(confidences, correctness)
    mce           = compute_mce(confidences, correctness)
    results["__all__"] = {"ece": ece, "mce": mce, "n": len(predictions)}

    print(f"\n  Global ECE : {ece:.4f}")
    print(f"  Global MCE : {mce:.4f}")

    # Global reliability diagram
    plot_reliability_diagram(
        bin_data, ece,
        title    = f"{domain} — Global Reliability Diagram",
        save_path= os.path.join(FIGURES_DIR, f"{domain.lower()}_reliability_global.png"),
        color    = COLORS[0],
    )
    print(f"  Saved → {domain.lower()}_reliability_global.png")

    # Confidence distribution
    plot_confidence_distribution(
        predictions, confidences, domain,
        save_path=os.path.join(FIGURES_DIR, f"{domain.lower()}_confidence_dist.png"),
    )

    # ── Subgroup ECE ──────────────────────────────────────────────────────────
    for axis_key, axis_name in SUBGROUP_AXES:
        print(f"\n  Subgroup: {axis_name}")

        groups = defaultdict(list)
        for i, ex in enumerate(predictions):
            groups[ex[axis_key]].append(i)

        subgroup_results = {}
        for label, indices in sorted(groups.items()):
            if len(indices) < 20:
                print(f"    {label:<20} skipped (n={len(indices)} < 20)")
                continue

            c = confidences[indices]
            a = correctness[indices]
            g_ece, g_bin_data = compute_ece(c, a)
            g_mce             = compute_mce(c, a)

            subgroup_results[label] = {
                "ece"     : g_ece,
                "mce"     : g_mce,
                "n"       : len(indices),
                "accuracy": float(a.mean()),
                "mean_conf": float(c.mean()),
                "bin_data": g_bin_data,
            }

            print(f"    {label:<20} n={len(indices):<6} ECE={g_ece:.4f}  acc={a.mean():.3f}  conf={c.mean():.3f}")

        results[axis_key] = subgroup_results

        # Subgroup reliability diagram grid
        plot_subgroup_reliability(
            subgroup_results, axis_name, domain,
            save_path=os.path.join(FIGURES_DIR, f"{domain.lower()}_reliability_{axis_key}.png"),
        )

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print_section("Calibration Analysis")

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
        predictions = json.load(open(path))
        results     = analyse_dataset(predictions, domain)
        all_results[domain] = results

    # ── Save metrics ──────────────────────────────────────────────────────────
    # Strip bin_data before saving (too large)
    saveable = {}
    for domain, domain_results in all_results.items():
        saveable[domain] = {}
        for key, val in domain_results.items():
            if key == "__all__":
                saveable[domain][key] = val
            else:
                saveable[domain][key] = {
                    label: {k: v for k, v in metrics.items() if k != "bin_data"}
                    for label, metrics in val.items()
                }

    out_path = os.path.join(RESULTS_DIR, "calibration_metrics.json")
    with open(out_path, "w") as f:
        json.dump(saveable, f, indent=2)

    print_section("Done")
    print(f"  Metrics saved → {out_path}")
    print(f"  Figures saved → {FIGURES_DIR}/")
    print(f"  Next step     : python src/risk_coverage.py\n")


if __name__ == "__main__":
    main()