from llama_cpp import Llama

def main():
    path = "models/Qwen3.5-4B-Q8_0.gguf"
    print(f"Loading {path}...")
    
    # Load with flash_attn=True exactly as your runner does
    llm = Llama(model_path=path, n_ctx=1024, flash_attn=True, verbose=False)

    prompt = (
        "Below is a long document. Answer the question using only information "
        "found in the document. Be concise.\n\n"
        "Document:\nThe secret treasury was moved to Obsidian in 1892.\n\n"
        "Question: Where was the secret treasury moved in 1892?\n"
        "Answer:"
    )

    print("\n--- TEST 1: Your current NeedleBench stop tokens ---")
    out1 = llm.create_completion(prompt, max_tokens=50, stop=["\n", "Question:", "Document:"])
    print(f"Generated: {repr(out1['choices'][0]['text'])}")

    print("\n--- TEST 2: Without the \\n stop token ---")
    out2 = llm.create_completion(prompt, max_tokens=50, stop=["Question:", "Document:", "<|im_end|>"])
    print(f"Generated: {repr(out2['choices'][0]['text'])}")

    print("\n--- TEST 3: Without Flash Attention (if the above are gibberish) ---")
    llm_no_flash = Llama(model_path=path, n_ctx=1024, flash_attn=False, verbose=False)
    out3 = llm_no_flash.create_completion(prompt, max_tokens=50, stop=["Question:", "Document:", "<|im_end|>"])
    print(f"Generated: {repr(out3['choices'][0]['text'])}")

if __name__ == "__main__":
    main()