"""
run_llm_inference.py
Runs Llama-3.2-3B-Instruct on a stratified sample of SQuAD validation.
Extracts token logprob confidence + verbalized uncertainty per example.
Saves predictions in the same schema as run_inference.py outputs.

Features:
  - Stratified sampling by question type (mirrors subgroup_tagger.py logic)
  - Checkpointing every CHECKPOINT_EVERY examples — safe to resume after crash
  - Output schema matches existing extractive model prediction files

Usage (Kaggle):
  - Set HF_TOKEN via Kaggle secrets before running
  - Run cells top to bottom
  - Output: outputs/predictions/llama_squad_predictions.json
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID         = "meta-llama/Llama-3.2-3B-Instruct"
SAMPLE_SIZE      = 3000          # total stratified sample
MAX_NEW_TOKENS   = 50            # max tokens for answer generation
VERBAL_TOKENS    = 5             # max tokens for verbalized confidence
CHECKPOINT_EVERY = 100           # save progress every N examples
OUTPUT_DIR       = os.path.join("outputs", "predictions")
CHECKPOINT_PATH  = os.path.join(OUTPUT_DIR, "llama_squad_checkpoint.json")
FINAL_PATH       = os.path.join(OUTPUT_DIR, "llama_squad_predictions.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Question Type (mirrors subgroup_tagger.py exactly) ────────────────────────

def get_question_type(question: str) -> str:
    q = question.strip().lower()
    if "highlight the parts" in q or "related to" in q:
        return "cuad_clause"
    first_word = q.split()[0] if q.split() else ""
    mapping = {
        "what" : "What", "which": "What",
        "who"  : "Who",  "whose": "Who",
        "when" : "When",
        "where": "Where",
        "how"  : "How",
        "why"  : "Why",
        "is"   : "YesNo", "are"  : "YesNo", "was"  : "YesNo",
        "were" : "YesNo", "did"  : "YesNo", "does" : "YesNo",
        "do"   : "YesNo", "can"  : "YesNo", "could": "YesNo",
        "would": "YesNo",
    }
    return mapping.get(first_word, "Other")


# ── Stratified Sampler ────────────────────────────────────────────────────────

def stratified_sample(dataset, total_size):
    """
    Sample `total_size` examples from dataset, stratified by question type.
    YesNo is the smallest subgroup (60 examples) — take all of them.
    All other types sampled proportionally to fill the remaining quota.
    Returns a flat list of examples.
    """
    examples = list(dataset)

    # Group by question type
    groups = defaultdict(list)
    for ex in examples:
        qtype = get_question_type(ex["question"])
        groups[qtype].append(ex)

    print("\n  Full dataset question type distribution:")
    for qtype, exs in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"    {qtype:<20} {len(exs):>6}")

    # Take all YesNo examples first (smallest subgroup)
    sampled          = {}
    yesno_all        = groups.get("YesNo", [])
    sampled["YesNo"] = yesno_all
    remaining_quota  = total_size - len(yesno_all)

    # Proportional allocation for remaining types
    other_types  = {k: v for k, v in groups.items() if k != "YesNo"}
    total_others = sum(len(v) for v in other_types.values())

    for qtype, exs in other_types.items():
        proportion     = len(exs) / total_others
        n              = round(proportion * remaining_quota)
        n              = min(n, len(exs))
        sampled[qtype] = exs[:n]

    # Flatten and report
    flat = []
    print("\n  Stratified sample breakdown:")
    for qtype, exs in sorted(sampled.items(), key=lambda x: -len(x[1])):
        print(f"    {qtype:<20} {len(exs):>6}")
        flat.extend(exs)

    print(f"\n  Total sampled: {len(flat)} examples")
    return flat


# ── Load Model ────────────────────────────────────────────────────────────────

def load_model(hf_token):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)

    print("Loading model (3-5 minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded ✓")
    return tokenizer, model


# ── Prompt Builders ───────────────────────────────────────────────────────────

def build_extraction_prompt(question, context, tokenizer):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise question answering assistant. "
                "Answer the question using only text from the passage. "
                "Copy the answer span exactly as it appears. "
                "If the passage does not contain the answer, "
                "reply with exactly one word: unanswerable"
            )
        },
        {
            "role": "user",
            "content": f"Passage: {context}\n\nQuestion: {question}\n\nAnswer:"
        }
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_verbalized_prompt(question, context, pred_answer, tokenizer):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a calibration assistant. "
                "Rate confidence in an answer from 0 to 100. "
                "Reply with ONLY a number between 0 and 100, nothing else."
            )
        },
        {
            "role": "user",
            "content": (
                f"Passage: {context}\n\n"
                f"Question: {question}\n\n"
                f"Answer given: {pred_answer}\n\n"
                "Confidence (0-100):"
            )
        }
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── Inference Helpers ─────────────────────────────────────────────────────────

def run_extraction(prompt, tokenizer, model):
    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids  = outputs.sequences[0][input_len:]
    pred_answer    = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    token_logprobs = []
    for i, scores in enumerate(outputs.scores):
        log_probs = F.log_softmax(scores[0], dim=-1)
        token_id  = generated_ids[i].item()
        token_logprobs.append(log_probs[token_id].item())

    mean_logprob = float(np.mean(token_logprobs)) if token_logprobs else 0.0
    return pred_answer, mean_logprob, token_logprobs


def run_verbalized(prompt, tokenizer, model):
    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=VERBAL_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0][input_len:]
    raw_text      = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    try:
        number = int(raw_text.split()[0].strip(".").strip("%"))
        number = max(0, min(100, number))
        return round(number / 100.0, 4)
    except (ValueError, IndexError):
        return None


# ── is_correct ────────────────────────────────────────────────────────────────

def compute_is_correct(pred_answer, gold_answers, is_answerable):
    pred = pred_answer.lower().strip()
    if not is_answerable:
        return pred == "unanswerable"
    for gold in gold_answers:
        g = gold.lower().strip()
        if pred == g or pred in g or g in pred:
            return True
    return False


# ── Checkpointing ─────────────────────────────────────────────────────────────

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            results = json.load(f)
        print(f"  Checkpoint found — resuming from example {len(results)}")
        return results
    return []


def save_checkpoint(results):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f)


# ── Main Inference Loop ───────────────────────────────────────────────────────

def run_inference(examples, tokenizer, model):
    results   = load_checkpoint()
    done_ids  = {r["id"] for r in results}
    remaining = [ex for ex in examples if ex["id"] not in done_ids]

    print(f"  {len(results)} already done, {len(remaining)} remaining\n")

    failed = 0

    for i, ex in enumerate(tqdm(remaining, desc="LLM Inference")):
        question      = ex["question"]
        context       = ex["context"]
        gold_answers  = ex["answers"]["text"]
        is_answerable = len(gold_answers) > 0

        try:
            ext_prompt              = build_extraction_prompt(question, context, tokenizer)
            pred_answer, mean_logprob, token_logprobs = run_extraction(
                ext_prompt, tokenizer, model
            )

            verb_prompt     = build_verbalized_prompt(question, context, pred_answer, tokenizer)
            verbalized_conf = run_verbalized(verb_prompt, tokenizer, model)

            is_correct = compute_is_correct(pred_answer, gold_answers, is_answerable)

            results.append({
                # Schema matching extractive model prediction files
                "id"              : ex["id"],
                "question"        : question,
                "gold_answers"    : gold_answers,
                "is_answerable"   : is_answerable,
                "pred_answer"     : pred_answer,
                "confidence"      : mean_logprob,
                "second_score"    : None,
                "delta"           : None,
                "null_score"      : None,
                # LLM-specific extras
                "verbalized_conf" : verbalized_conf,
                "token_logprobs"  : token_logprobs,
                "is_correct"      : is_correct,
                "model"           : "llama-3.2-3b-instruct",
            })

        except Exception as e:
            failed += 1
            print(f"\n  Failed on id={ex['id']}: {e}")
            results.append({
                "id"              : ex["id"],
                "question"        : question,
                "gold_answers"    : gold_answers,
                "is_answerable"   : is_answerable,
                "pred_answer"     : "",
                "confidence"      : None,
                "second_score"    : None,
                "delta"           : None,
                "null_score"      : None,
                "verbalized_conf" : None,
                "token_logprobs"  : [],
                "is_correct"      : False,
                "model"           : "llama-3.2-3b-instruct",
            })

        # Checkpoint every N examples
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(results)
            print(f"\n  Checkpoint saved at {len(results)} examples")

    save_checkpoint(results)
    print(f"\n  Done. {len(results)} total | {failed} failures")
    return results


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login

    secrets  = UserSecretsClient()
    HF_TOKEN = secrets.get_secret("HF_TOKEN")
    login(token=HF_TOKEN)
    print("HF login ✓")

    # Load model
    tokenizer, model = load_model(HF_TOKEN)

    # Load + stratify SQuAD validation
    print("\nLoading SQuAD 2.0...")
    squad    = load_dataset("rajpurkar/squad_v2")["validation"]
    examples = stratified_sample(squad, SAMPLE_SIZE)

    # Full run
    print(f"\nStarting full run on {len(examples)} examples...")
    results = run_inference(examples, tokenizer, model)

    with open(FINAL_PATH, "w") as f:
        json.dump(results, f)
    print(f"\n✓ Final predictions saved → {FINAL_PATH}")