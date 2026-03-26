"""
GSM8K benchmark implementation using llama-cpp-python directly.
250 samples, extracts numerical answer with multi-strategy regex, compares to ground truth.
Uses few-shot prompting with #### format to ensure consistent answer extraction across models.
"""
import json
import re
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from llama_cpp import Llama


# Few-shot examples demonstrating the expected #### format
FEWSHOT_EXAMPLES = """Solve this math problem step by step, then end with #### followed by the final numerical answer.

Question: Janet has 3 apples and buys 4 more. How many apples does she have?

Answer:
Janet starts with 3 apples.
She buys 4 more apples.
3 + 4 = 7
#### 7

Question: A box contains 12 books. If 5 books are removed, how many remain?

Answer:
The box starts with 12 books.
5 books are removed.
12 - 5 = 7
#### 7

Question: {question}

Answer:"""


def extract_answer(text: str) -> float | None:
    """Extract final numerical answer using multi-strategy approach."""
    # Clean text: normalize whitespace, handle commas
    text = re.sub(r'\s+', ' ', text).strip()
    text_clean = text.replace(",", "")
    
    # Strategy 1: GSM8K standard format #### <number>
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text_clean)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # Strategy 2: LaTeX boxed format \boxed{<number>}
    match = re.search(r"\\boxed\{(-?\d+(?:\.\d+)?)\}", text_clean)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # Strategy 3: XML answer tags <answer>...</answer>
    match = re.search(r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>", text_clean)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # Strategy 4: Look for answer after keywords (case-insensitive)
    # Look in the last 300 characters where final answer typically appears
    last_part = text_clean[-300:] if len(text_clean) > 300 else text_clean
    
    keyword_patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is[\s:]+(-?\d+(?:\.\d+)?)",
        r"(?:total|sum|altogether|result|finally)[:\s]+(-?\d+(?:\.\d+)?)",
        r"(?:equals?|=?)[\s:]+(-?\d+(?:\.\d+)?)",
    ]
    
    for pattern in keyword_patterns:
        match = re.search(pattern, last_part, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    
    # Strategy 5: Last number in text (fallback)
    # Find all numbers and take the last one
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text_clean)
    if numbers:
        # Filter out small numbers that might be step numbers (e.g., "Step 1")
        # and look for numbers that appear after the main reasoning
        for num in reversed(numbers):
            try:
                val = float(num)
                # Skip very small integers that might be step indicators
                if val >= 0 or len(numbers) == 1:
                    return val
            except ValueError:
                continue
    
    return None


def evaluate_gsm8k(model_path: str, limit: int = 250, n_ctx: int = 2048) -> dict:
    """Run GSM8K evaluation on a model."""
    
    # Load model
    print(f"Loading model: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,  # Full GPU offloading
        n_ctx=n_ctx,
        verbose=False,
    )
    
    # Load GSM8K dataset
    print(f"Loading GSM8K dataset (limit: {limit})...")
    dataset = load_dataset("gsm8k", "main", split="test")
    samples = dataset.select(range(min(limit, len(dataset))))
    
    results = []
    correct = 0
    
    for i, sample in enumerate(tqdm(samples, desc="GSM8K")):
        question = sample["question"]
        # Ground truth is after ####
        ground_truth_text = sample["answer"].split("####")[-1].strip()
        try:
            ground_truth = float(ground_truth_text.replace(",", ""))
        except ValueError:
            continue
        
        # Generate answer using few-shot prompt format
        prompt = FEWSHOT_EXAMPLES.format(question=question)
        
        output = llm.create_completion(
            prompt=prompt,
            max_tokens=512,
            temperature=0.0,
            stop=["<|im_start|>", "Question:"],
        )
        
        generated_text = output["choices"][0]["text"]
        predicted = extract_answer(generated_text)
        
        is_correct = predicted is not None and abs(predicted - ground_truth) < 1e-3
        if is_correct:
            correct += 1
        
        results.append({
            "index": i,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "generated_text": generated_text,
            "correct": is_correct,
        })
    
    accuracy = correct / len(results) if results else 0.0
    
    return {
        "model": model_path,
        "benchmark": "gsm8k",
        "limit": limit,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "results": results,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GSM8K benchmark")
    parser.add_argument("model_path", help="Path to GGUF model")
    parser.add_argument("--limit", type=int, default=250, help="Number of samples")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context size")
    
    args = parser.parse_args()
    
    results = evaluate_gsm8k(args.model_path, args.limit, args.n_ctx)
    
    # Save results
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    print(f"\nAccuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")


if __name__ == "__main__":
    main()
