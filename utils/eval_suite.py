"""Evaluation Suite — Automated benchmarking, A/B comparison, metrics.

Features:
  - Perplexity calculation
  - BLEU score (n-gram overlap)
  - ROUGE-like score (recall-oriented)
  - A/B model comparison
  - Eval logging and history
"""

import os
import json
import math
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_perplexity(losses: List[float]) -> float:
    """Calculate perplexity from a list of cross-entropy losses."""
    if not losses:
        return float('inf')
    avg_loss = sum(losses) / len(losses)
    return math.exp(min(avg_loss, 100))  # Cap to avoid overflow


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Get n-gram counts from a token list."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> Dict[str, float]:
    """Calculate BLEU score between reference and hypothesis.

    Returns dict with bleu-1 through bleu-N and overall score.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens or not ref_tokens:
        return {f"bleu-{n}": 0.0 for n in range(1, max_n + 1)}

    scores = {}
    precisions = []

    for n in range(1, max_n + 1):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        hyp_ngrams = _get_ngrams(hyp_tokens, n)

        if not hyp_ngrams:
            scores[f"bleu-{n}"] = 0.0
            precisions.append(0.0)
            continue

        # Clipped counts
        clipped = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count

        precision = clipped / total if total > 0 else 0.0
        scores[f"bleu-{n}"] = round(precision, 4)
        precisions.append(precision)

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        scores["bleu"] = round(bp * math.exp(log_avg), 4)
    else:
        scores["bleu"] = 0.0

    return scores


def rouge_score(reference: str, hypothesis: str) -> Dict[str, float]:
    """Calculate ROUGE-L like score (longest common subsequence based).

    Returns dict with precision, recall, f1.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # LCS
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    precision = lcs_length / n if n > 0 else 0.0
    recall = lcs_length / m if m > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER)."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return round(dp[m][n] / max(m, 1), 4)


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

def evaluate_model_on_dataset(
    model, tokenizer, test_data: List[Dict],
    config: dict, device: str,
    max_samples: int = 50,
) -> Dict[str, Any]:
    """Run full evaluation on a test dataset.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        test_data: List of {"prompt": ..., "response": ...}
        config: Generation config
        device: Device string
        max_samples: Max number of samples to evaluate

    Returns:
        Dict with metrics and individual results
    """
    import torch
    from services.chat import generate_response

    model.eval()
    results = []
    all_bleu = []
    all_rouge = []
    start_time = time.time()

    samples = test_data[:max_samples]
    print(f"\n  Evaluating on {len(samples)} samples...")

    for i, item in enumerate(samples):
        prompt = item["prompt"]
        reference = item["response"]

        try:
            with torch.inference_mode():
                hypothesis = generate_response(model, tokenizer, prompt, config, device)
        except Exception as e:
            hypothesis = f"[Error: {e}]"

        b = bleu_score(reference, hypothesis)
        r = rouge_score(reference, hypothesis)

        all_bleu.append(b["bleu"])
        all_rouge.append(r["f1"])

        results.append({
            "prompt": prompt,
            "reference": reference,
            "hypothesis": hypothesis,
            "bleu": b,
            "rouge": r,
        })

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(samples)} done...")

    elapsed = time.time() - start_time

    summary = {
        "num_samples": len(samples),
        "avg_bleu": round(sum(all_bleu) / len(all_bleu), 4) if all_bleu else 0,
        "avg_rouge_f1": round(sum(all_rouge) / len(all_rouge), 4) if all_rouge else 0,
        "eval_time_seconds": round(elapsed, 2),
        "results": results,
    }

    print(f"\n  Evaluation Results:")
    print("  " + "-" * 40)
    print(f"  Samples:    {summary['num_samples']}")
    print(f"  Avg BLEU:   {summary['avg_bleu']:.4f}")
    print(f"  Avg ROUGE:  {summary['avg_rouge_f1']:.4f}")
    print(f"  Time:       {elapsed:.1f}s")

    return summary


# ---------------------------------------------------------------------------
# A/B Comparison
# ---------------------------------------------------------------------------

def compare_models(
    model_a, model_b, tokenizer_a, tokenizer_b,
    prompts: List[str], config: dict, device: str,
    name_a: str = "Model A", name_b: str = "Model B",
) -> Dict:
    """Compare two models side by side on the same prompts."""
    import torch
    from services.chat import generate_response

    results = []
    scores_a, scores_b = [], []

    print(f"\n  A/B Comparison: {name_a} vs {name_b}")
    print(f"  Testing {len(prompts)} prompts...\n")

    for i, prompt in enumerate(prompts):
        try:
            with torch.inference_mode():
                resp_a = generate_response(model_a, tokenizer_a, prompt, config, device)
                resp_b = generate_response(model_b, tokenizer_b, prompt, config, device)
        except Exception as e:
            resp_a = resp_b = f"[Error: {e}]"

        # Score responses (length, diversity as proxies)
        score_a = len(resp_a.split()) * (len(set(resp_a.lower().split())) / max(len(resp_a.split()), 1))
        score_b = len(resp_b.split()) * (len(set(resp_b.lower().split())) / max(len(resp_b.split()), 1))
        scores_a.append(score_a)
        scores_b.append(score_b)

        results.append({
            "prompt": prompt,
            "response_a": resp_a,
            "response_b": resp_b,
            "score_a": round(score_a, 2),
            "score_b": round(score_b, 2),
        })

        print(f"  [{i + 1}] Prompt: {prompt[:60]}...")
        print(f"      {name_a}: {resp_a[:80]}...")
        print(f"      {name_b}: {resp_b[:80]}...")
        print()

    avg_a = sum(scores_a) / len(scores_a) if scores_a else 0
    avg_b = sum(scores_b) / len(scores_b) if scores_b else 0
    winner = name_a if avg_a > avg_b else name_b

    summary = {
        "name_a": name_a, "name_b": name_b,
        "avg_score_a": round(avg_a, 2),
        "avg_score_b": round(avg_b, 2),
        "winner": winner,
        "results": results,
    }

    print(f"  Summary: {name_a}={avg_a:.2f} vs {name_b}={avg_b:.2f} → {winner} wins")
    return summary


# ---------------------------------------------------------------------------
# Eval Logging
# ---------------------------------------------------------------------------

EVAL_LOG_PATH = "data/eval_history.json"


def log_eval(eval_name: str, metrics: Dict, model_name: str = ""):
    """Log evaluation results to history."""
    os.makedirs(os.path.dirname(EVAL_LOG_PATH), exist_ok=True)

    history = []
    if os.path.exists(EVAL_LOG_PATH):
        with open(EVAL_LOG_PATH, 'r') as f:
            history = json.load(f)

    entry = {
        "name": eval_name,
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": {k: v for k, v in metrics.items() if k != "results"},
    }
    history.append(entry)

    with open(EVAL_LOG_PATH, 'w') as f:
        json.dump(history, f, indent=2)


def get_eval_history() -> List[Dict]:
    """Load evaluation history."""
    if os.path.exists(EVAL_LOG_PATH):
        with open(EVAL_LOG_PATH, 'r') as f:
            return json.load(f)
    return []


# ---------------------------------------------------------------------------
# Interactive Eval Suite
# ---------------------------------------------------------------------------

def interactive_eval():
    """Interactive evaluation suite."""
    print("\n" + "=" * 55)
    print("       Evaluation Suite")
    print("=" * 55)

    while True:
        print("\n  Options:")
        print("  1  Quick metrics (BLEU, ROUGE on text pair)")
        print("  2  View eval history")
        print("  3  Dataset quality metrics")
        print("  0  Back")
        print("\n  (Full model eval requires loaded model — use from training pipeline)")

        try:
            choice = input("\n  eval>> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice in ('0', 'back', 'quit', 'q'):
            break

        try:
            if choice == '1':
                ref = input("  Reference text: ").strip()
                hyp = input("  Hypothesis text: ").strip()
                if ref and hyp:
                    b = bleu_score(ref, hyp)
                    r = rouge_score(ref, hyp)
                    w = word_error_rate(ref, hyp)
                    print(f"\n  BLEU:  {b}")
                    print(f"  ROUGE: {r}")
                    print(f"  WER:   {w}")

            elif choice == '2':
                history = get_eval_history()
                if history:
                    print(f"\n  Evaluation History ({len(history)} entries):")
                    print("  " + "-" * 60)
                    for h in history[-10:]:
                        metrics_str = ", ".join(f"{k}={v}" for k, v in h.get("metrics", {}).items()
                                                if isinstance(v, (int, float)))
                        print(f"  {h['timestamp'][:19]}  {h.get('model', ''):<15} {metrics_str}")
                else:
                    print("  No eval history found.")

            elif choice == '3':
                path = input("  Dataset path [data/test.json]: ").strip() or "data/test.json"
                with open(path, 'r') as f:
                    data = json.load(f)
                # Quick stats
                prompts = [d["prompt"] for d in data]
                responses = [d["response"] for d in data]
                avg_p_len = sum(len(p.split()) for p in prompts) / len(prompts) if prompts else 0
                avg_r_len = sum(len(r.split()) for r in responses) / len(responses) if responses else 0
                print(f"\n  Dataset: {path} ({len(data)} entries)")
                print(f"  Avg prompt length:   {avg_p_len:.1f} words")
                print(f"  Avg response length: {avg_r_len:.1f} words")
                all_words = " ".join(prompts + responses).lower().split()
                vocab = len(set(all_words))
                print(f"  Vocabulary size:     {vocab}")
                print(f"  Total tokens:        {len(all_words)}")

        except (KeyboardInterrupt, EOFError):
            continue
        except Exception as e:
            print(f"  Error: {e}")
