"""
Analysis script for benchmark results.
Generates comparison plots for the paper figures.
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

# Results directories
RESULTS_DIR = Path("results/completed")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def load_all_results():
    """Load all completed benchmark results."""
    results = defaultdict(dict)
    
    if not RESULTS_DIR.exists():
        print(f"No results found in {RESULTS_DIR}")
        return results
    
    for json_file in RESULTS_DIR.glob("*_*.json"):
        # Parse filename: MODEL_BENCHMARK.json
        parts = json_file.stem.split("_")
        if len(parts) >= 2:
            model = "_".join(parts[:-1])  # Everything except last part
            benchmark = parts[-1]
            
            try:
                with open(json_file) as f:
                    data = json.load(f)
                results[benchmark][model] = data
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    return results


def parse_model_config(model_name: str):
    """Parse model name into (architecture, quant_level)."""
    # Examples:
    # Qwen3-4B-Q8 -> (Qwen3, Q8)
    # Qwen3.5-4B-Q4 -> (Qwen3.5, Q4)
    
    if "Qwen3.5" in model_name:
        arch = "Qwen3.5 (GDN Hybrid)"
        base = "Qwen3.5"
    elif "Qwen3" in model_name:
        arch = "Qwen3 (Pure Transformer)"
        base = "Qwen3"
    else:
        arch = model_name
        base = model_name
    
    # Extract quant level
    if "Q8" in model_name:
        quant = "Q8"
    elif "Q4" in model_name:
        quant = "Q4"
    elif "Q3" in model_name:
        quant = "Q3"
    else:
        quant = "unknown"
    
    return arch, base, quant


def plot_perplexity(results):
    """Plot perplexity comparison (Figure 1)."""
    if "perplexity" not in results or not results["perplexity"]:
        print("No perplexity results found")
        return
    
    perplexity_data = results["perplexity"]
    
    # Organize data
    data = []
    for model_name, result in perplexity_data.items():
        arch, base, quant = parse_model_config(model_name)
        ppl = result.get("perplexity", float('nan'))
        data.append({
            "Model": base,
            "Architecture": arch,
            "Quantization": quant,
            "Perplexity": ppl
        })
    
    if not data:
        print("No valid perplexity data")
        return
    
    # Create plot
    fig, ax = plt.subplots()
    
    quant_order = ["Q8", "Q4", "Q3"]
    models = sorted(set(d["Model"] for d in data))
    
    x = np.arange(len(quant_order))
    width = 0.35
    
    for i, model in enumerate(models):
        values = []
        for q in quant_order:
            val = next((d["Perplexity"] for d in data 
                       if d["Model"] == model and d["Quantization"] == q), None)
            values.append(val if val is not None else 0)
        
        offset = width * (i - len(models)/2 + 0.5)
        ax.bar(x + offset, values, width, label=model)
    
    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Figure 1: Perplexity Degradation by Quantization Level")
    ax.set_xticks(x)
    ax.set_xticklabels(quant_order)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure1_perplexity.png", dpi=150)
    plt.savefig(FIGURES_DIR / "figure1_perplexity.pdf")
    print(f"Saved: {FIGURES_DIR / 'figure1_perplexity.png'}")
    plt.close()


def plot_gsm8k(results):
    """Plot GSM8K accuracy comparison (Figure 2)."""
    if "gsm8k" not in results or not results["gsm8k"]:
        print("No GSM8K results found")
        return
    
    gsm8k_data = results["gsm8k"]
    
    data = []
    for model_name, result in gsm8k_data.items():
        arch, base, quant = parse_model_config(model_name)
        acc = result.get("accuracy", 0) * 100
        data.append({
            "Model": base,
            "Architecture": arch,
            "Quantization": quant,
            "Accuracy": acc
        })
    
    if not data:
        print("No valid GSM8K data")
        return
    
    fig, ax = plt.subplots()
    
    quant_order = ["Q8", "Q4", "Q3"]
    models = sorted(set(d["Model"] for d in data))
    
    x = np.arange(len(quant_order))
    width = 0.35
    
    for i, model in enumerate(models):
        values = []
        for q in quant_order:
            val = next((d["Accuracy"] for d in data 
                       if d["Model"] == model and d["Quantization"] == q), 0)
            values.append(val)
        
        offset = width * (i - len(models)/2 + 0.5)
        ax.bar(x + offset, values, width, label=model)
    
    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Figure 2: GSM8K Math Reasoning Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(quant_order)
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure2_gsm8k.png", dpi=150)
    plt.savefig(FIGURES_DIR / "figure2_gsm8k.pdf")
    print(f"Saved: {FIGURES_DIR / 'figure2_gsm8k.png'}")
    plt.close()


def plot_niah(results):
    """Plot NIAH (Needle-in-a-Haystack) results (Figure 3 - main claim)."""
    # Use niah (random) if available, fall back to niah_ablation
    niah_key = "niah" if "niah" in results and results["niah"] else "niah_ablation"
    if niah_key not in results or not results[niah_key]:
        print("No NIAH results found")
        return
    
    niah_data = results[niah_key]
    
    # Organize by model and context length
    data = defaultdict(lambda: defaultdict(dict))
    
    for model_name, result in niah_data.items():
        arch, base, quant = parse_model_config(model_name)
        
        for ctx_result in result.get("results", []):
            ctx_len = ctx_result.get("context_length", 0)
            acc = ctx_result.get("accuracy", 0) * 100
            
            data[base][quant][ctx_len] = acc
    
    if not data:
        print("No valid NIAH data")
        return
    
    # Create subplot for each context length
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    context_lengths = [4096, 8192]
    quant_order = ["Q8", "Q4", "Q3"]
    
    for idx, ctx_len in enumerate(context_lengths):
        ax = axes[idx]
        
        models = sorted(data.keys())
        x = np.arange(len(quant_order))
        width = 0.35
        
        for i, model in enumerate(models):
            values = []
            for q in quant_order:
                val = data[model].get(q, {}).get(ctx_len, 0)
                values.append(val)
            
            offset = width * (i - len(models)/2 + 0.5)
            ax.bar(x + offset, values, width, label=model)
        
        ax.set_xlabel("Quantization Level")
        ax.set_ylabel("Retrieval Accuracy (%)")
        ax.set_title(f"Context Length: {ctx_len} tokens")
        ax.set_xticks(x)
        ax.set_xticklabels(quant_order)
        ax.legend()
        ax.set_ylim(0, 100)
    
    fig.suptitle("Figure 3: Long-Context Retrieval (NIAH) - Main Experimental Result", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure3_niah.png", dpi=150)
    plt.savefig(FIGURES_DIR / "figure3_niah.pdf")
    print(f"Saved: {FIGURES_DIR / 'figure3_niah.png'}")
    plt.close()


def plot_niah_ablation(results):
    """Plot NIAH position ablation results (Figure 4 - secondary analysis)."""
    if "niah_ablation" not in results or not results["niah_ablation"]:
        print("No NIAH ablation results found")
        return
    
    niah_abl_data = results["niah_ablation"]
    
    # Organize data by model, quant, context length, and depth
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for model_name, result in niah_abl_data.items():
        arch, base, quant = parse_model_config(model_name)
        
        for ctx_result in result.get("results", []):
            ctx_len = ctx_result.get("context_length", 0)
            acc_by_depth = ctx_result.get("accuracy_by_depth", {})
            
            for depth_str, acc in acc_by_depth.items():
                data[base][quant][ctx_len][depth_str] = acc * 100
    
    if not data:
        print("No valid NIAH ablation data")
        return
    
    # Create subplot for each context length
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    context_lengths = [4096, 8192]
    depth_order = ["10%", "25%", "50%", "75%", "90%"]
    
    for idx, ctx_len in enumerate(context_lengths):
        ax = axes[idx]
        
        models = sorted(data.keys())
        x = np.arange(len(depth_order))
        width = 0.35
        
        for i, model in enumerate(models):
            # Average across quant levels for cleaner view, or pick one
            # Let's show Q4 as the most interesting case
            quant = "Q4"
            values = []
            for d in depth_order:
                val = data[model].get(quant, {}).get(ctx_len, {}).get(d, 0)
                values.append(val)
            
            if any(v > 0 for v in values):  # Only plot if we have data
                offset = width * (i - len(models)/2 + 0.5)
                ax.bar(x + offset, values, width, label=f"{model} ({quant})")
        
        ax.set_xlabel("Needle Depth in Document")
        ax.set_ylabel("Retrieval Accuracy (%)")
        ax.set_title(f"Context Length: {ctx_len} tokens")
        ax.set_xticks(x)
        ax.set_xticklabels(depth_order)
        ax.legend()
        ax.set_ylim(0, 100)
    
    fig.suptitle("Figure 4: Position-Dependent Retrieval (NIAH Ablation) - Secondary Analysis", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure4_niah_ablation.png", dpi=150)
    plt.savefig(FIGURES_DIR / "figure4_niah_ablation.pdf")
    print(f"Saved: {FIGURES_DIR / 'figure4_niah_ablation.png'}")
    plt.close()


def print_summary_table(results):
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Perplexity
    if "perplexity" in results and results["perplexity"]:
        print("\nPerplexity (lower is better):")
        print("-" * 60)
        for model_name in sorted(results["perplexity"].keys()):
            result = results["perplexity"][model_name]
            ppl = result.get("perplexity", float('nan'))
            print(f"  {model_name:30s}: {ppl:.2f}")
    
    # GSM8K
    if "gsm8k" in results and results["gsm8k"]:
        print("\nGSM8K Accuracy (higher is better):")
        print("-" * 60)
        for model_name in sorted(results["gsm8k"].keys()):
            result = results["gsm8k"][model_name]
            acc = result.get("accuracy", 0) * 100
            correct = result.get("correct", 0)
            total = result.get("total", 0)
            print(f"  {model_name:30s}: {acc:.1f}% ({correct}/{total})")
    
    # NIAH
    if "niah" in results and results["niah"]:
        print("\nNIAH Retrieval Accuracy (higher is better):")
        print("-" * 60)
        for model_name in sorted(results["niah"].keys()):
            result = results["niah"][model_name]
            print(f"  {model_name}:")
            for ctx_result in result.get("results", []):
                ctx_len = ctx_result.get("context_length", 0)
                acc = ctx_result.get("accuracy", 0) * 100
                correct = ctx_result.get("correct", 0)
                total = ctx_result.get("total", 0)
                print(f"    {ctx_len} tokens: {acc:.1f}% ({correct}/{total})")
    
    print("\n" + "=" * 80)


def main():
    print("Loading benchmark results...")
    results = load_all_results()
    
    if not any(results.values()):
        print("No results found. Run benchmarks first.")
        return
    
    # Print summary
    print_summary_table(results)
    
    # Generate figures
    print("\nGenerating figures...")
    plot_perplexity(results)
    plot_gsm8k(results)
    plot_niah(results)
    plot_niah_ablation(results)  # New ablation figure
    
    print(f"\nFigures saved to: {FIGURES_DIR}/")
    print("  - figure1_perplexity.png")
    print("  - figure2_gsm8k.png")
    print("  - figure3_niah.png (main result)")
    print("  - figure4_niah_ablation.png (position ablation)")


if __name__ == "__main__":
    main()
