"""
Needle-in-a-Haystack (NIAH) 2-Hop Stress Test.
Forces Relational Reasoning by splitting the fact across two needles.
Needle A: Object -> Person | Needle B: Person -> City.
Includes 10 Distractor chains (20 total distractor needles).
"""
import json
import random
from pathlib import Path
from tqdm import tqdm
from llama_cpp import Llama
from datasets import load_dataset

# Set seed for reproducible NIAH needle positions
random.seed(42)

_HAYSTACK_CORPUS = ""

def create_haystack(length_chars: int) -> str:
    """Create a complex, real-world text haystack using WikiText-103."""
    global _HAYSTACK_CORPUS
    if not _HAYSTACK_CORPUS:
        try:
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            valid_texts = [row["text"].strip() for row in dataset.select(range(20000)) if len(row["text"].strip()) > 80]
            _HAYSTACK_CORPUS = " ".join(valid_texts)
        except Exception as e:
            _HAYSTACK_CORPUS = "The quick brown fox jumps over the lazy dog. " * 2000
    haystack = _HAYSTACK_CORPUS
    while len(haystack) < length_chars:
        haystack += haystack
    return haystack[:length_chars]

def get_2hop_data():
    return {
        "objects": ["Amulet", "Chronometer", "Manuscript", "Scepter", "Tapestry", "Compass", "Astrolabe", "Reliquary", "Statue", "Casket", "Scroll", "Engraving", "Fossil", "Meteorite", "Medallion"],
        "people": ["Dr. Aris", "Prof. Elara", "Captain Zephyr", "Chancellor Vane", "Oracle Pythia", "Warden Kael", "Senator Thorne", "Archivist Silas", "Baroness Myra", "Explorer Jax", "Scholar Juno", "Artisan Theo", "Magus Orion", "Guildmaster Bram", "Commander Nyx"],
        "cities": ["Aethelgard", "Beryllos", "Cythera", "Drakonias", "Eos", "Fayril", "Gondolin", "Helios", "Ithaca", "Jovian", "Kaldoran", "Lumeria", "Mithralis", "Nyxos", "Ophidia"]
    }

def create_2hop_needle_set(num_distractor_chains=10):
    """Creates a 2-hop chain and several distractor chains."""
    data = get_2hop_data()
    
    # Sample unique elements for all chains
    objs = random.sample(data["objects"], num_distractor_chains + 1)
    ppl = random.sample(data["people"], num_distractor_chains + 1)
    cities = random.sample(data["cities"], num_distractor_chains + 1)
    
    target_obj = objs[0]
    target_person = ppl[0]
    target_city = cities[0]
    
    question = f"In what city is the person who commissioned the {target_obj} currently residing?"
    
    needles = []
    # Create True Chain
    needles.append({"text": f"The historical {target_obj} was commissioned by {target_person} for the royal collection.", "is_target": True})
    needles.append({"text": f"According to local records, {target_person} is currently residing in the city of {target_city}.", "is_target": True})
    
    # Create Distractor Chains
    for i in range(1, num_distractor_chains + 1):
        needles.append({"text": f"The historical {objs[i]} was commissioned by {ppl[i]} for the royal collection.", "is_target": False})
        needles.append({"text": f"According to local records, {ppl[i]} is currently residing in the city of {cities[i]}.", "is_target": False})
    
    random.shuffle(needles)
    return needles, target_city, question

def evaluate_niah(
    model_path: str,
    context_lengths: list[int] = [16000],
    num_positions: int = 30,
    n_ctx: int = 16384,
    fixed_positions: bool = False,
    n_gpu_layers: int = -1,
    n_batch: int = 256,
) -> dict:
    """Run NIAH benchmark with 2-Hop Relational Reasoning."""
    
    print(f"Loading model: {model_path}")
    try:
        llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_batch=n_batch, n_ctx=n_ctx, flash_attn=True, verbose=False)
    except Exception as e:
        raise RuntimeError(f"FATAL OOM: {e}")

    all_results = []
    
    for ctx_len in context_lengths:
        print(f"\nTesting 2-Hop NIAH: {ctx_len}")
        haystack = create_haystack(ctx_len * 4)
        haystack_tokens = llm.tokenize(haystack.encode())
        haystack_tokens = haystack_tokens[:ctx_len - 1000] # Leave room for 20+ needles
        
        correct = 0
        trials = []
        
        test_positions = [0.10, 0.25, 0.50, 0.75, 0.90] if fixed_positions else [random.uniform(0.05, 0.95) for _ in range(num_positions)]
        if fixed_positions:
            test_positions = test_positions * (num_positions // len(test_positions))

        for target_depth in tqdm(test_positions, desc=f"NIAH {ctx_len}"):
            needle_set, expected, question = create_2hop_needle_set(num_distractor_chains=10)
            
            temp_tokens = list(haystack_tokens)
            for n in needle_set:
                n_tokens = llm.tokenize(n["text"].encode())
                # If target, place near the designated depth. If distractor, place randomly.
                depth = target_depth + random.uniform(-0.02, 0.02) if n["is_target"] else random.uniform(0.05, 0.95)
                pos = int(len(temp_tokens) * max(0.01, min(0.99, depth)))
                temp_tokens = temp_tokens[:pos] + n_tokens + temp_tokens[pos:]
            
            context = llm.detokenize(temp_tokens).decode('utf-8', errors='ignore')
            prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
            
            output = llm.create_completion(prompt=prompt, max_tokens=30, temperature=0.0, stop=["\n", "."])
            generated = output["choices"][0]["text"].strip()
            
            is_correct = expected.lower() in generated.lower()
            if is_correct: correct += 1
            
            trials.append({"depth": target_depth, "needle": expected, "question": question, "predicted": generated, "correct": is_correct})
            
        accuracy = correct / len(trials) if trials else 0.0
        
        # Calculate accuracy by depth
        accuracy_by_depth = {}
        for d in [0.10, 0.25, 0.50, 0.75, 0.90]:
            matches = [t for t in trials if abs(t["depth"] - d) < 0.05]
            if matches:
                accuracy_by_depth[f"{int(d*100)}%"] = sum(1 for m in matches if m["correct"]) / len(matches)

        all_results.append({
            "context_length": ctx_len,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(trials),
            "positions": trials,
            "accuracy_by_depth": accuracy_by_depth
        })
    
    return {"model": model_path, "benchmark": "niah", "results": all_results}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[16000])
    parser.add_argument("--num-positions", type=int, default=30)
    parser.add_argument("--output", type=str)
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--fixed-positions", action="store_true")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-batch", type=int, default=256)
    args = parser.parse_args()
    
    res = evaluate_niah(args.model_path, args.context_lengths, args.num_positions, args.n_ctx, args.fixed_positions, args.n_gpu_layers, args.n_batch)
    
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f: json.dump(res, f, indent=2)

if __name__ == "__main__": 
    main()