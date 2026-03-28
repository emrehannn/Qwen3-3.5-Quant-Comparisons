import json
import random
import sys
from pathlib import Path

# Add src/ to path so we can import helpers
sys.path.append(str(Path("src").resolve()))
from needlebench import (
    load_needlebench_subset, 
    build_haystack_with_multiple_needles, 
    extract_structured_answer,
    composite_retrieval_score
)
from llama_cpp import Llama

def get_trials(filepath):
    with open(filepath) as f: root = json.load(f)
    trials = []
    for entry in root["tasks"]["M-RS"]:
        ctx = entry.get("context_length", 0)
        for trial in entry.get("trials", []):
            trial["context_length"] = ctx
            trials.append(trial)
    return trials

print("Analyzing past JSON logs to find failure intersections...")
q8_trials = get_trials("results/completed/Qwen3-4B-Q8_needlebench.json")
u3_trials = get_trials("results/completed/Qwen3-4B-UD-Q3_K_XL_needlebench.json")

q8_dict = {f"{x['context_length']}_{x.get('depth_percent', x.get('depth', 0))}_{x.get('question', '')[:40]}": x for x in q8_trials}
u3_dict = {f"{x['context_length']}_{x.get('depth_percent', x.get('depth', 0))}_{x.get('question', '')[:40]}": x for x in u3_trials}

matched_targets = []
for k, q8_trial in q8_dict.items():
    if k in u3_dict:
        u3_trial = u3_dict[k]
        q8_correct = q8_trial.get("correct", False)
        u3_correct = u3_trial.get("correct", False)
        q8_text = q8_trial.get("predicted", q8_trial.get("predicted_text", "")).strip()
        
        if not q8_correct and u3_correct and "NOT FOUND" in q8_text and q8_trial["context_length"] == 4096:
            matched_targets.append(q8_trial)
            if len(matched_targets) == 3: # Keep it fast
                break

haystack_ds = load_needlebench_subset("en_haystack_texts", split="test")
haystack_texts = [s["text"] for s in haystack_ds if s.get("text")]
reasoning_ds = load_needlebench_subset("multi_needle_reasoning_needle", split="test")

_PROMPT_ABLATED = (
    "Below is a long document. Read it carefully and answer the question.\n"
    "Document:\n{context}\n\n"
    "Question: {question}\n"
    "Instructions: You must wrap your final, concise answer inside <answer> and </answer> tags.\n"
    "Response:"
)

def evaluate_model(model_name, model_path, targets):
    print(f"\n{'='*60}\nEvaluating ablation on {model_name}\n{'='*60}")
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_batch=256, n_ctx=4096, flash_attn=True, verbose=False)
    
    for idx, target in enumerate(targets):
        question = target["question"]
        expected = target.get("expected", target.get("ground_truth", ""))
        depth    = target.get("depth_percent", target.get("depth", 0)) / 100.0
        ctx_len  = target["context_length"]
        target_tokens = ctx_len - 400
        
        ds_sample = next((s for s in reasoning_ds if s.get("question", "") == question), None)
        if not ds_sample: continue
            
        derivations = ds_sample.get("derivations", [])
        if isinstance(derivations, str):
            try: derivations = json.loads(derivations)
            except: derivations = [derivations]
        if not isinstance(derivations, list) or len(derivations) == 0:
            derivations = [expected]
            
        rng = random.Random(42)  
        context = build_haystack_with_multiple_needles(
            llm, haystack_texts, derivations, target_tokens, depth, rng)
            
        output = llm.create_completion(
            prompt=_PROMPT_ABLATED.format(context=context, question=question),
            max_tokens=150, temperature=0.0,
            stop=["Question:", "Document:", "<|im_end|>"])
            
        raw_generated = output["choices"][0]["text"].strip()
        generated     = extract_structured_answer(raw_generated)
        score         = composite_retrieval_score(generated, expected)
        
        print(f"\n[Target {idx+1}] Context: {ctx_len} tokens | Depth: {int(depth*100)}%")
        print(f"Goal Expected : {expected}")
        print(f"Actual Result : {generated}")
        print(f"Passed Score  : {'YES (>=0.5)' if score >= 0.5 else 'NO (<0.5)'} (Raw Score: {score:.2f})")

evaluate_model("Qwen3-4B Q8", "models/Qwen3-4B-Instruct-2507-Q8_0.gguf", matched_targets)
evaluate_model("Qwen3-4B UD-Q3", "models/Qwen3-4B-Instruct-2507-UD-Q3_K_XL.gguf", matched_targets)
print("\nABLATION COMPLETE")
