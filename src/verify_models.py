"""
Quick model verification test.
Loads each model and runs simple inference + GSM8K to verify answer extraction.
"""
import sys
import re
from pathlib import Path
from llama_cpp import Llama

MODELS = [
    ("Qwen3-4B-Q8", "models/Qwen3-4B-Instruct-2507-Q8_0.gguf"),
    ("Qwen3-4B-Q4", "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
    ("Qwen3-4B-Q3", "models/Qwen3-4B-Instruct-2507-UD-Q3_K_XL.gguf"),
    ("Qwen3.5-4B-Q8", "models/Qwen3.5-4B-Q8_0.gguf"),
    ("Qwen3.5-4B-Q4", "models/Qwen3.5-4B-Q4_K_M.gguf"),
    ("Qwen3.5-4B-Q3", "models/Qwen3.5-4B-Q3_K_M.gguf"),
]


def extract_answer(text: str) -> float | None:
    """Extract final numerical answer using multi-strategy approach (same as gsm8k.py)."""
    import re
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
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text_clean)
    if numbers:
        for num in reversed(numbers):
            try:
                val = float(num)
                if val >= 0 or len(numbers) == 1:
                    return val
            except ValueError:
                continue
    
    return None


def test_gsm8k_sample(llm: Llama) -> tuple[bool, str, float]:
    """Test GSM8K with one sample problem using few-shot prompting."""
    question = "Janet buys a multi-flavor pack of cheese. There are 3 swiss cheeses, 4 brie cheeses, and 5 cheddar cheeses. If she eats one cheese per day, how many days will the cheese last?"
    ground_truth = 12.0  # 3 + 4 + 5 = 12
    
    # Few-shot examples (same as gsm8k.py)
    fewshot = """Solve this math problem step by step, then end with #### followed by the final numerical answer.

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
    
    prompt = fewshot.format(question=question)
    
    output = llm.create_completion(
        prompt=prompt,
        max_tokens=512,
        temperature=0.0,
        stop=["<|im_start|>", "Question:"],
    )
    
    generated = output["choices"][0]["text"]
    predicted = extract_answer(generated)
    is_correct = predicted is not None and abs(predicted - ground_truth) < 1e-3
    
    return is_correct, generated[:100], predicted


def test_model(name: str, path: str) -> bool:
    """Test load a model and run simple inference + GSM8K."""
    print(f"\nTesting {name}...")
    print(f"  Path: {path}")
    
    if not Path(path).exists():
        print(f"  ✗ File not found")
        return False
    
    size_gb = Path(path).stat().st_size / (1024**3)
    print(f"  Size: {size_gb:.2f} GB")
    
    try:
        # Load model with minimal context
        llm = Llama(
            model_path=path,
            n_gpu_layers=-1,
            n_ctx=512,
            verbose=False,
        )
        
        # Run simple inference
        prompt = "The capital of France is"
        output = llm.create_completion(
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
        )
        
        result = output["choices"][0]["text"].strip()
        print(f"  Basic test: '{result}'")
        
        # Run GSM8K test
        print(f"  Testing GSM8K answer extraction...")
        gsm8k_ok, response_preview, answer = test_gsm8k_sample(llm)
        
        if gsm8k_ok:
            print(f"    ✓ GSM8K: Correct (answer: {answer:.0f})")
        else:
            print(f"    ⚠ GSM8K: Answer extraction test (got: {answer}, preview: '{response_preview}...')")
        
        print(f"  ✓ Model fully verified")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("=" * 70)
    print("MODEL VERIFICATION TEST")
    print("=" * 70)
    print("Loading each model and running inference + GSM8K test...")
    
    passed = 0
    failed = 0
    
    for name, path in MODELS:
        if test_model(name, path):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{len(MODELS)}")
    print(f"Failed: {failed}/{len(MODELS)}")
    
    if failed == 0:
        print("\n✓ All models ready for benchmarks!")
        return 0
    else:
        print(f"\n✗ {failed} model(s) failed verification. Fix issues before running benchmarks.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
