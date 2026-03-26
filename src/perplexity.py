"""
WikiText-103 perplexity benchmark.
Forward-pass only, measures raw information loss per quantization level.
"""
import json
import math
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from llama_cpp import Llama


import numpy as np


def evaluate_perplexity(model_path: str, limit: int = 1000, n_ctx: int = 2048) -> dict:
    """Calculate perplexity on WikiText-103."""
    
    print(f"Loading model: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        verbose=False,
        logits_all=True,  # Enable logits output
    )
    
    # Load WikiText-103
    print(f"Loading WikiText-103 (limit: {limit} examples)...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    
    total_loss = 0.0
    total_tokens = 0
    
    for i, example in enumerate(tqdm(dataset, desc="Perplexity", total=limit)):
        if i >= limit:
            break
            
        text = example["text"].strip()
        if len(text) < 50:  # Skip very short examples
            continue
        
        # Tokenize
        tokens = llm.tokenize(text.encode())
        if len(tokens) < 2:
            continue
        
        # Process in chunks to avoid OOM
        chunk_size = n_ctx - 1  # Leave room for target token
        for chunk_start in range(0, len(tokens) - 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size + 1, len(tokens))
            chunk_tokens = tokens[chunk_start:chunk_end]
            
            if len(chunk_tokens) < 2:
                continue
            
            # Get logits by evaluating context
            # Reset state for each chunk
            llm.reset()
            llm.eval(chunk_tokens[:-1])
            
            # Get logits for last position
            logits = llm._scores  # Get raw logits from last evaluation
            if logits is None or len(logits) == 0:
                continue
            
            # Get target token for each position - vectorized computation
            num_positions = min(len(chunk_tokens) - 1, len(logits))
            if num_positions > 0:
                target_tokens = np.array(chunk_tokens[1:num_positions + 1])
                
                # Vectorized softmax: (num_positions, vocab_size)
                logits_array = np.array(logits[:num_positions])
                max_logits = np.max(logits_array, axis=1, keepdims=True)
                exp_logits = np.exp(logits_array - max_logits)
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                
                # Get probabilities for target tokens
                target_probs = probs[np.arange(num_positions), target_tokens]
                valid_mask = target_probs > 0
                
                # Compute cross-entropy loss
                total_loss += np.sum(-np.log(target_probs[valid_mask] + 1e-10))
                total_tokens += np.sum(valid_mask)
        
        # Periodic cleanup
        if i % 100 == 0:
            import gc
            gc.collect()
    
    perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
    
    return {
        "model": model_path,
        "benchmark": "wikitext103",
        "limit": limit,
        "perplexity": float(perplexity),
        "total_tokens": int(total_tokens),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run WikiText-103 perplexity")
    parser.add_argument("model_path", help="Path to GGUF model")
    parser.add_argument("--limit", type=int, default=1000, help="Number of examples")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context size")
    
    args = parser.parse_args()
    
    results = evaluate_perplexity(args.model_path, args.limit, args.n_ctx)
    
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    print(f"\nPerplexity: {results['perplexity']:.2f}")


if __name__ == "__main__":
    main()
