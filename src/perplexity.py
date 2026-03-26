"""
WikiText-103 perplexity benchmark.
Forward-pass only, measures raw information loss per quantization level.

Requires logits_all=True at model load time. Uses llama_cpp's low-level
eval() API to get per-token logits, then computes cross-entropy loss.
Uses llm.scores (public) with fallback to llm._scores for older versions.
"""
import gc
import json
import math
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from llama_cpp import Llama
import numpy as np


def get_scores(llm):
    """Get logit scores using public API with fallback to private."""
    # Try public attribute first (llama-cpp-python >= 0.2.x)
    scores = getattr(llm, "scores", None)
    if scores is not None and len(scores) > 0:
        return scores
    # Fallback to private attribute
    scores = getattr(llm, "_scores", None)
    if scores is not None and len(scores) > 0:
        return scores
    return None


def evaluate_perplexity(model_path: str, limit: int = 1000, n_ctx: int = 2048) -> dict:
    """
    Calculate perplexity on WikiText-103.

    Loads model with logits_all=True, evaluates token chunks with llm.eval(),
    reads logits from llm.scores, computes softmax cross-entropy loss.
    Processes in non-overlapping chunks of n_ctx-1 tokens.
    """

    print(f"Loading model: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        logits_all=True,   # REQUIRED: enables per-token logits from eval()
        flash_attn=True,
        verbose=False,
    )

    print(f"Loading WikiText-103 (limit: {limit} examples)...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    total_nll   = 0.0   # sum of negative log-likelihoods
    total_tokens = 0

    for i, example in enumerate(tqdm(dataset, desc="Perplexity", total=limit)):
        if i >= limit:
            break

        text = example["text"].strip()
        if len(text) < 50:
            continue

        tokens = llm.tokenize(text.encode())
        if len(tokens) < 2:
            continue

        # Process in non-overlapping chunks
        chunk_size = n_ctx - 1   # leave 1 slot for the final target token
        for chunk_start in range(0, len(tokens) - 1, chunk_size):
            chunk = tokens[chunk_start : chunk_start + chunk_size + 1]
            if len(chunk) < 2:
                continue

            # Feed all tokens except the last one as context
            context = chunk[:-1]
            targets = chunk[1:]    # targets[i] = token predicted after context[i]

            # Reset KV cache and evaluate
            llm.reset()
            llm.eval(context)

            # Get logits matrix: shape (len(context), vocab_size)
            scores = get_scores(llm)
            if scores is None:
                print("  [warn] Could not retrieve logits — skipping chunk")
                continue

            n_pos = min(len(context), len(scores), len(targets))
            if n_pos == 0:
                continue

            logits  = np.array(scores[:n_pos], dtype=np.float32)  # (n_pos, vocab)
            targets_arr = np.array(targets[:n_pos])

            # Numerically stable softmax → log-probabilities
            logits -= logits.max(axis=1, keepdims=True)
            log_sum_exp = np.log(np.exp(logits).sum(axis=1))
            # log P(target_i | context_i) = logits[i, target_i] - log_sum_exp[i]
            target_logits = logits[np.arange(n_pos), targets_arr]
            log_probs     = target_logits - log_sum_exp

            total_nll    += -log_probs.sum()
            total_tokens += n_pos

        if i % 100 == 0:
            gc.collect()

    if total_tokens == 0:
        print("WARNING: No tokens collected — logits API may not be working")
        perplexity = float("inf")
    else:
        perplexity = math.exp(total_nll / total_tokens)

    print(f"  total_tokens={total_tokens}, nll={total_nll:.2f}, ppl={perplexity:.2f}")

    return {
        "model":        model_path,
        "benchmark":    "wikitext103",
        "limit":        limit,
        "perplexity":   float(perplexity),
        "total_tokens": int(total_tokens),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run WikiText-103 perplexity")
    parser.add_argument("model_path", help="Path to GGUF model")
    parser.add_argument("--limit",  type=int, default=1000, help="Number of examples")
    parser.add_argument("--output", type=str,               help="Output JSON file")
    parser.add_argument("--n-ctx",  type=int, default=2048, help="Context size")

    args = parser.parse_args()

    results = evaluate_perplexity(args.model_path, args.limit, args.n_ctx)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    print(f"\nPerplexity: {results['perplexity']:.2f}  ({results['total_tokens']} tokens)")


if __name__ == "__main__":
    main()