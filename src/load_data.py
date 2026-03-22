"""
load_data.py
Loads SQuAD2.0 and CUAD from HuggingFace, runs sanity checks,
and saves 100-example samples to outputs/ for inspection.

Usage: python src/load_data.py
"""

import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ── Output dir ────────────────────────────────────────────────────────────────
SAMPLE_OUTPUT_DIR = os.path.join("outputs", "samples")
os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def sanity_check_squad(dataset):
    print_section("SQuAD 2.0 Sanity Check")

    for split in ["train", "validation"]:
        ds = dataset[split]
        total = len(ds)

        # Answerable vs unanswerable
        answerable   = sum(1 for ex in ds if len(ex["answers"]["text"]) > 0)
        unanswerable = total - answerable

        print(f"\n  Split: {split}")
        print(f"    Total examples     : {total}")
        print(f"    Answerable         : {answerable} ({100*answerable/total:.1f}%)")
        print(f"    Unanswerable       : {unanswerable} ({100*unanswerable/total:.1f}%)")
        print(f"    Fields             : {list(ds.features.keys())}")

    # Spot check one example
    ex = dataset["validation"][0]
    print(f"\n  Sample example (validation[0]):")
    print(f"    Question : {ex['question']}")
    print(f"    Context  : {ex['context'][:120]}...")
    print(f"    Answers  : {ex['answers']}")


def sanity_check_cuad(dataset):
    print_section("CUAD Sanity Check")

    for split in ["train", "test"]:
        if split not in dataset:
            print(f"\n  Split '{split}' not found — skipping")
            continue

        ds = dataset[split]
        total = len(ds)

        # CUAD answers field is same structure as SQuAD
        answerable   = sum(1 for ex in tqdm(ds, desc=f"  Counting {split}", leave=False)
                          if len(ex["answers"]["text"]) > 0)
        unanswerable = total - answerable

        print(f"\n  Split: {split}")
        print(f"    Total examples     : {total}")
        print(f"    Answerable         : {answerable} ({100*answerable/total:.1f}%)")
        print(f"    Unanswerable       : {unanswerable} ({100*unanswerable/total:.1f}%)")
        print(f"    Fields             : {list(ds.features.keys())}")

    # Spot check one example
    ex = dataset["test"][0]
    print(f"\n  Sample example (test[0]):")
    print(f"    Question : {ex['question']}")
    print(f"    Context  : {ex['context'][:120]}...")
    print(f"    Answers  : {ex['answers']}")


def save_sample(dataset, split, sample_size, filename):
    """Save first `sample_size` examples as a flat JSON file for inspection."""
    ds = dataset[split]
    samples = []

    for ex in list(ds)[:sample_size]:
        samples.append({
            "id"      : ex.get("id", ""),
            "question": ex["question"],
            "context" : ex["context"][:300],   # truncate for readability
            "answers" : ex["answers"],
        })

    out_path = os.path.join(SAMPLE_OUTPUT_DIR, filename)
    with open(out_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\n  Saved {sample_size} samples → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── SQuAD 2.0 ─────────────────────────────────────────────────────────────
    print_section("Loading SQuAD 2.0")
    squad = load_dataset("rajpurkar/squad_v2")
    print("  Loaded SQuAD 2.0 ✓")

    sanity_check_squad(squad)
    save_sample(squad, "validation", 100, "squad_v2_sample.json")

    # ── CUAD ──────────────────────────────────────────────────────────────────
    print_section("Loading CUAD")
    cuad = load_dataset("theatticusproject/cuad")
    print("  Loaded CUAD ✓")

    sanity_check_cuad(cuad)
    save_sample(cuad, "test", 100, "cuad_sample.json")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_section("Done")
    print("  Both datasets loaded and verified.")
    print("  Samples saved to outputs/samples/")
    print("  Next step: python src/run_inference.py\n")


if __name__ == "__main__":
    main()
