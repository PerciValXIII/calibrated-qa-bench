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

    available_splits = list(dataset.keys())
    print(f"\n  Available splits: {available_splits}")

    for split in available_splits:
        ds = dataset[split]
        total = len(ds)

        answerable   = sum(1 for ex in tqdm(ds, desc=f"  Counting {split}", leave=False)
                          if len(ex["answers"]["text"]) > 0)
        unanswerable = total - answerable

        print(f"\n  Split: {split}")
        print(f"    Total examples     : {total}")
        print(f"    Answerable         : {answerable} ({100*answerable/total:.1f}%)")
        print(f"    Unanswerable       : {unanswerable} ({100*unanswerable/total:.1f}%)")
        print(f"    Fields             : {list(ds.features.keys())}")

    # Spot check first available split
    first_split = available_splits[0]
    ex = dataset[first_split][0]
    print(f"\n  Sample example ({first_split}[0]):")
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
    # Load CUAD from local zip file at data/cuad_dataset/data.zip
    import zipfile
    from datasets import Dataset, DatasetDict

    cuad_zip_path = os.path.join("data", "cuad_dataset", "data.zip")
    print(f"  Reading CUAD from {cuad_zip_path}...")

    def parse_squad_json(file_obj):
        """Parse a SQuAD-format JSON file into a list of flat rows."""
        raw = json.load(file_obj)
        rows = []
        for article in raw["data"]:
            title = article.get("title", "")
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    answers = qa.get("answers", [])
                    rows.append({
                        "id"      : qa["id"],
                        "title"   : title,
                        "context" : context,
                        "question": qa["question"],
                        "answers" : {
                            "text"        : [a["text"] for a in answers],
                            "answer_start" : [a["answer_start"] for a in answers],
                        }
                    })
        return rows

    with zipfile.ZipFile(cuad_zip_path) as z:
        available = z.namelist()
        print(f"  Files in zip: {available}")

        # Load train and test splits
        train_file = next(n for n in available if "train_separate_questions" in n)
        test_file  = next(n for n in available if n.endswith("test.json"))

        with z.open(train_file) as f:
            train_rows = parse_squad_json(f)
        with z.open(test_file) as f:
            test_rows = parse_squad_json(f)

    cuad = DatasetDict({
        "train": Dataset.from_list(train_rows),
        "test" : Dataset.from_list(test_rows),
    })
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