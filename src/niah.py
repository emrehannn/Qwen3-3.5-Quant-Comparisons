"""
Needle-in-a-Haystack (NIAH) benchmark for long-context retrieval.
Tests at 4k and 8k context lengths with varying needle positions.
"""
import json
import random
from pathlib import Path
from tqdm import tqdm
from llama_cpp import Llama

# Set seed for reproducible NIAH needle positions
random.seed(42)


def create_haystack(length_chars: int) -> str:
    """Create a repetitive text haystack of approximately given length."""
    # Use repetitive sentences to fill context without semantic meaning
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump.",
        "Sphinx of black quartz, judge my vow.",
        "Two driven jocks help fax my big quiz.",
    ]
    haystack = ""
    while len(haystack) < length_chars:
        haystack += " " + random.choice(sentences)
    return haystack


def create_needle() -> tuple[str, str]:
    """Create a unique needle with a specific fact to retrieve."""
    # Generate a random code/number that the model needs to retrieve
    code = f"NIAH-{random.randint(1000, 9999)}"
    needle = f"The secret code is: {code}. Remember this code."
    return needle, code


def evaluate_niah(
    model_path: str,
    context_lengths: list[int] = [4096, 8192],
    num_positions: int = 30,
    n_ctx: int = 16384,
    fixed_positions: bool = False,
) -> dict:
    """Run NIAH benchmark at specified context lengths.
    
    Args:
        fixed_positions: If True, test at fixed depths (10%, 25%, 50%, 75%, 90%)
                        If False, use random placement (standard NIAH)
    """
    
    print(f"Loading model: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        verbose=False,
    )
    
    all_results = []
    
    for ctx_len in context_lengths:
        print(f"\nTesting context length: {ctx_len}")
        
        # Create haystack (approximate character count to token count ratio ~4:1)
        haystack_chars = ctx_len * 4
        haystack = create_haystack(haystack_chars)
        
        # Convert to tokens to get accurate length
        haystack_tokens = llm.tokenize(haystack.encode())
        
        # Truncate to exact context length minus room for needle and prompt
        available_space = ctx_len - 100  # Reserve space
        if len(haystack_tokens) > available_space:
            haystack_tokens = haystack_tokens[:available_space]
            haystack = llm.detokenize(haystack_tokens).decode('utf-8', errors='ignore')
        
        correct = 0
        positions_tested = []
        
        # Determine test positions
        if fixed_positions:
            # Fixed depths for ablation: 10%, 25%, 50%, 75%, 90%
            depths = [0.10, 0.25, 0.50, 0.75, 0.90]
            trials_per_depth = max(1, num_positions // len(depths))
            test_positions = []
            for depth in depths:
                for _ in range(trials_per_depth):
                    # Add small jitter (±2%) to avoid exact structural boundaries
                    jitter = random.uniform(-0.02, 0.02)
                    test_positions.append(min(0.95, max(0.05, depth + jitter)))
            random.shuffle(test_positions)  # Shuffle to avoid order effects
        else:
            # Random placement (standard NIAH)
            test_positions = [random.uniform(0.05, 0.95) for _ in range(num_positions)]
        
        for depth in test_positions:
            needle, code = create_needle()
            insert_pos = int(len(haystack_tokens) * depth)
            
            # Insert needle into haystack
            needle_tokens = llm.tokenize(needle.encode())
            context_tokens = (
                haystack_tokens[:insert_pos] + 
                needle_tokens + 
                haystack_tokens[insert_pos:]
            )
            context = llm.detokenize(context_tokens).decode('utf-8', errors='ignore')
            
            # Query the model
            prompt = f"{context}\n\nQuestion: What is the secret code mentioned in the text above?\nAnswer:"
            
            output = llm.create_completion(
                prompt=prompt,
                max_tokens=50,
                temperature=0.0,
                stop=["\n"],
            )
            
            generated = output["choices"][0]["text"].strip()
            is_correct = code in generated
            
            if is_correct:
                correct += 1
            
            positions_tested.append({
                "depth": depth,
                "needle": code,
                "predicted": generated,
                "correct": is_correct,
            })
        
        # Calculate overall accuracy
        accuracy = correct / len(positions_tested) if positions_tested else 0.0
        
        # Calculate accuracy by depth bins if using fixed positions
        accuracy_by_depth = {}
        if fixed_positions:
            from collections import defaultdict
            depth_results = defaultdict(lambda: {'correct': 0, 'total': 0})
            for pos_info in positions_tested:
                # Round to nearest standard depth
                depth = pos_info['depth']
                closest_depth = min([0.10, 0.25, 0.50, 0.75, 0.90], 
                                   key=lambda x: abs(x - depth))
                depth_results[closest_depth]['total'] += 1
                if pos_info['correct']:
                    depth_results[closest_depth]['correct'] += 1
            
            for d in [0.10, 0.25, 0.50, 0.75, 0.90]:
                if depth_results[d]['total'] > 0:
                    accuracy_by_depth[f"{int(d*100)}%"] = (
                        depth_results[d]['correct'] / depth_results[d]['total']
                    )
        
        result_entry = {
            "context_length": ctx_len,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(positions_tested),
            "positions": positions_tested,
        }
        
        if accuracy_by_depth:
            result_entry["accuracy_by_depth"] = accuracy_by_depth
            depth_str = ", ".join([f"{k}: {v:.1%}" for k, v in accuracy_by_depth.items()])
            print(f"  Accuracy at {ctx_len}: {accuracy:.1%} ({correct}/{len(positions_tested)})")
            print(f"    By depth: {depth_str}")
        else:
            print(f"  Accuracy at {ctx_len}: {accuracy:.1%} ({correct}/{len(positions_tested)})")
        
        all_results.append(result_entry)
    
    return {
        "model": model_path,
        "benchmark": "niah",
        "fixed_positions": fixed_positions,
        "context_lengths_tested": context_lengths,
        "results": all_results,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run NIAH benchmark")
    parser.add_argument("model_path", help="Path to GGUF model")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--num-positions", type=int, default=30)
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--n-ctx", type=int, default=16384, help="Model max context")
    parser.add_argument("--fixed-positions", action="store_true", help="Use fixed depths (10%, 25%, 50%, 75%, 90%) instead of random")
    
    args = parser.parse_args()
    
    results = evaluate_niah(
        args.model_path,
        args.context_lengths,
        args.num_positions,
        args.n_ctx,
        args.fixed_positions,
    )
    
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
