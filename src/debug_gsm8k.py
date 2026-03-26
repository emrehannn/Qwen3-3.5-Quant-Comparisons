"""
Debug GSM8K output to see what models are actually generating.
"""
import sys
import re
from pathlib import Path
from llama_cpp import Llama

# Test just one sample with detailed output
QUESTION = "Janet buys a multi-flavor pack of cheese. There are 3 swiss cheeses, 4 brie cheeses, and 5 cheddar cheeses. If she eats one cheese per day, how many days will the cheese last?"

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


def test_model(name: str, path: str):
    print(f"\n{'='*70}")
    print(f"Testing {name}")
    print(f"{'='*70}")
    
    llm = Llama(
        model_path=path,
        n_gpu_layers=-1,
        n_ctx=512,
        verbose=False,
    )
    
    prompt = FEWSHOT_EXAMPLES.format(question=QUESTION)
    print(f"\nPrompt:\n{prompt[:200]}...")
    
    output = llm.create_completion(
        prompt=prompt,
        max_tokens=512,
        temperature=0.0,
        stop=["<|im_start|>", "<|im_end|>", "Question:"],
    )
    
    generated = output["choices"][0]["text"]
    print(f"\nGenerated text:\n{generated}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    # Test one Qwen3 and one Qwen3.5 model
    test_model("Qwen3-4B-Q8", "models/Qwen3-4B-Instruct-2507-Q8_0.gguf")
    test_model("Qwen3.5-4B-Q8", "models/Qwen3.5-4B-Q8_0.gguf")