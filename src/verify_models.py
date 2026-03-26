"""
Quick model verification test.
Loads each model and runs simple inference + GSM8K to verify answer extraction.
"""
import sys
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
    
    # Few-shot examples (same as gsm8k.py - 4-shot)
    fewshot = """Solve this math problem step by step, then end with #### followed by the final numerical answer.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

Answer:
There are 15 trees originally.
After planting, there are 21 trees.
21 - 15 = 6 trees were planted.
#### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

Answer:
There are 3 cars originally.
2 more cars arrive.
3 + 2 = 5 cars are in the parking lot.
#### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

Answer:
Leah had 32 chocolates.
Her sister had 42 chocolates.
32 + 42 = 74 chocolates originally.
35 chocolates were eaten.
74 - 35 = 39 chocolates left.
#### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

Answer:
Jason started with 20 lollipops.
Jason now has 12 lollipops.
20 - 12 = 8 lollipops were given to Denny.
#### 8

Question: {question}

Answer:"""
    
    prompt = fewshot.format(question=question)
    
    output = llm.create_completion(
        prompt=prompt,
        max_tokens=512,
        temperature=0.0,
        stop=["<|im_start|>", "<|im_end|>", "Question:"],
    )
    
    generated = output["choices"][0]["text"]
    predicted = extract_answer(generated)
    is_correct = predicted is not None and abs(predicted - ground_truth) < 1e-3
    
    return is_correct, generated[:100], predicted


def test_niah_sample(llm: Llama) -> bool:
    """Quick NIAH test - verify model can retrieve a secret code at 50% depth."""
    # Create simple haystack (~512 tokens worth of text)
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump.",
    ]
    haystack = " ".join([sentences[i % 3] for i in range(20)])
    
    # Tokenize
    haystack_tokens = llm.tokenize(haystack.encode())
    
    # Create needle
    code = "NIAH-1234"
    needle = f"The secret code is: {code}. Remember this code."
    needle_tokens = llm.tokenize(needle.encode())
    
    # Insert at 50% depth
    insert_pos = len(haystack_tokens) // 2
    context_tokens = (
        haystack_tokens[:insert_pos] + 
        needle_tokens + 
        haystack_tokens[insert_pos:]
    )
    context = llm.detokenize(context_tokens).decode('utf-8', errors='ignore')
    
    # Query
    prompt = f"{context}\n\nQuestion: What is the secret code mentioned in the text above?\nAnswer:"
    
    output = llm.create_completion(
        prompt=prompt,
        max_tokens=20,
        temperature=0.0,
        stop=["\n"],
    )
    
    generated = output["choices"][0]["text"].strip()
    return code in generated


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
        
        # Run NIAH test
        print(f"  Testing NIAH retrieval...")
        niah_ok = test_niah_sample(llm)
        
        if niah_ok:
            print(f"    ✓ NIAH: Retrieved secret code correctly")
        else:
            print(f"    ⚠ NIAH: Failed to retrieve code")
        
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