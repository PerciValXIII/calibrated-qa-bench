"""
run_inference.py
Runs extractive QA models on SQuAD validation and CUAD test sets.
Saves per-example predictions + confidence signals to outputs/predictions/.

Supported models (--model flag):
    roberta-base    → deepset/roberta-base-squad2      (default, already done)
    roberta-large   → deepset/roberta-large-squad2
    deberta         → deepset/deberta-v3-base-squad2

Usage:
    python src/run_inference.py --model roberta-large --full
    python src/run_inference.py --model deberta --full
    python src/run_inference.py --model roberta-base --full   # re-run base
"""

import os
import json
import argparse
import zipfile
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset, Dataset

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "roberta-base" : "deepset/roberta-base-squad2",
    "roberta-large": "deepset/roberta-large-squad2",
    "deberta"      : "deepset/deberta-v3-base-squad2",
}

# ── Config ────────────────────────────────────────────────────────────────────
MAX_EXAMPLES = 20
MAX_LENGTH   = 512
DOC_STRIDE   = 128
OUTPUT_DIR   = os.path.join("outputs", "predictions")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def get_confidence(start_logits, end_logits, top_k=5):
    # Use float64 for precision; clip to safe range to prevent inf/NaN
    # in arithmetic (roberta-large produces much larger logit magnitudes)
    start      = np.clip(np.array(start_logits, dtype=np.float64), -1e4, 1e4)
    end        = np.clip(np.array(end_logits,   dtype=np.float64), -1e4, 1e4)

    null_score = float(start[0] + end[0])

    candidates = []
    for i in range(min(top_k * 2, len(start))):
        for j in range(i, min(i + 50, len(end))):
            score = float(start[i] + end[j])
            if not np.isnan(score):
                candidates.append((score, i, j))
    candidates.sort(reverse=True)

    best_score   = candidates[0][0] if candidates else 0.0
    second_score = candidates[1][0] if len(candidates) > 1 else 0.0
    delta        = best_score - second_score

    # Final NaN guard — defensive fallback for any edge case
    best_score   = 0.0 if np.isnan(best_score)   else best_score
    second_score = 0.0 if np.isnan(second_score) else second_score
    delta        = 0.0 if np.isnan(delta)         else delta
    null_score   = 0.0 if np.isnan(null_score)    else null_score

    return best_score, second_score, delta, null_score


def run_inference_on_dataset(dataset, tokenizer, model, dataset_name, max_examples, use_float64=False):
    examples = list(dataset)
    if max_examples:
        examples = examples[:max_examples]

    # On CPU, roberta-large's attention kernel can produce NaN logits in float32.
    # Casting the entire model to float64 for the forward pass fixes this.
    if use_float64:
        model = model.double()
        print("  Note: Model cast to float64 for CPU stability (roberta-large).")

    results  = []
    nan_count = 0

    for ex in tqdm(examples, desc=f"  Inference on {dataset_name}"):
        question = ex["question"]
        context  = ex["context"]

        inputs = tokenizer(
            question,
            context,
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=DOC_STRIDE,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        offset_mapping = inputs.pop("offset_mapping")

        # Cast float inputs to match model dtype when using float64
        if use_float64:
            inputs = {
                k: v.double() if v.is_floating_point() else v
                for k, v in inputs.items()
            }

        with torch.no_grad():
            outputs = model(**inputs)

        # Always convert back to float32 numpy for storage
        start_logits = outputs.start_logits[0].float().numpy().tolist()
        end_logits   = outputs.end_logits[0].float().numpy().tolist()

        # Detect any residual NaNs — zero-fill and count
        has_nan = any(np.isnan(v) for v in start_logits) or \
                  any(np.isnan(v) for v in end_logits)
        if has_nan:
            nan_count   += 1
            start_logits = [0.0 if np.isnan(v) else v for v in start_logits]
            end_logits   = [0.0 if np.isnan(v) else v for v in end_logits]

        start_idx = int(np.argmax(start_logits))
        end_idx   = int(np.argmax(end_logits))

        offsets = offset_mapping[0].numpy()
        try:
            char_start  = int(offsets[start_idx][0])
            char_end    = int(offsets[end_idx][1])
            pred_answer = context[char_start:char_end] if char_start < char_end else ""
        except Exception:
            pred_answer = ""

        best_score, second_score, delta, null_score = get_confidence(
            start_logits, end_logits
        )

        gold_answers  = ex["answers"]["text"]
        is_answerable = len(gold_answers) > 0

        results.append({
            "id"           : ex["id"],
            "question"     : question,
            "gold_answers" : gold_answers,
            "is_answerable": is_answerable,
            "pred_answer"  : pred_answer,
            "confidence"   : best_score,
            "second_score" : second_score,
            "delta"        : delta,
            "null_score"   : null_score,
            "start_logits" : start_logits,
            "end_logits"   : end_logits,
        })

    if nan_count:
        print(f"  Warning: {nan_count} examples still had NaN logits after float64 cast — zero-filled.")
    else:
        print(f"  No NaN logits detected.")

    return results


def save_predictions(results, model_key, dataset_name):
    """Save to outputs/predictions/{model_key}_{dataset_name}_predictions.json"""
    filename = f"{model_key}_{dataset_name}_predictions.json"
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"  Saved {len(results)} predictions → {out_path}")
    return out_path


def load_cuad_local():
    cuad_zip_path = os.path.join("data", "cuad_dataset", "data.zip")

    def parse_squad_json(file_obj):
        raw  = json.load(file_obj)
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
        test_file = next(n for n in z.namelist() if n.endswith("test.json"))
        with z.open(test_file) as f:
            test_rows = parse_squad_json(f)

    return Dataset.from_list(test_rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(model_key, full_run):
    if model_key not in MODELS:
        print(f"  Unknown model '{model_key}'. Choose from: {list(MODELS.keys())}")
        return

    model_name = MODELS[model_key]
    max_ex     = None if full_run else MAX_EXAMPLES

    # roberta-large produces NaN logits on CPU in float32 due to attention
    # kernel behaviour — use float64 for the forward pass to fix this.
    use_float64 = (model_key == "roberta-large") and (not torch.cuda.is_available())

    print_section(f"Loading Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForQuestionAnswering.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    print(f"  Model key  : {model_key}")
    print(f"  HF name    : {model_name}")
    print(f"  Device     : {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Precision  : {'float64 (NaN fix)' if use_float64 else 'float32'}")
    print(f"  Examples   : {max_ex if max_ex else 'ALL (full run)'}")

    # ── SQuAD ─────────────────────────────────────────────────────────────────
    print_section("SQuAD 2.0 Inference")
    squad = load_dataset("rajpurkar/squad_v2")
    squad_results = run_inference_on_dataset(
        squad["validation"], tokenizer, model,
        dataset_name="SQuAD validation",
        max_examples=max_ex,
        use_float64=use_float64,
    )
    save_predictions(squad_results, model_key, "squad")

    # ── CUAD ──────────────────────────────────────────────────────────────────
    print_section("CUAD Inference")
    cuad_test    = load_cuad_local()
    cuad_results = run_inference_on_dataset(
        cuad_test, tokenizer, model,
        dataset_name="CUAD test",
        max_examples=max_ex,
        use_float64=use_float64,
    )
    save_predictions(cuad_results, model_key, "cuad")

    print_section("Done")
    print(f"  Model     : {model_key}")
    print(f"  SQuAD     : {len(squad_results)} predictions")
    print(f"  CUAD      : {len(cuad_results)} predictions")
    print(f"  Files     : outputs/predictions/{model_key}_squad_predictions.json")
    print(f"            : outputs/predictions/{model_key}_cuad_predictions.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="roberta-base",
        choices=list(MODELS.keys()),
        help="Which model to run inference with"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run on full datasets instead of 500-example sample"
    )
    args = parser.parse_args()
    main(model_key=args.model, full_run=args.full)