"""
Refusal Percentage Table Generator
Creates a formatted table of refusal percentages for all models and quantization types.
"""

import json
from pathlib import Path
from collections import defaultdict

# Directories
RESULTS_DIR = Path("results/completed")

# Model and quantization mappings
MODELS = ["Qwen3", "Qwen3.5"]
QUANTS = ["Q8", "Q4", "Q3", "UD-Q3"]

# File mapping
FILES = {
    ("Qwen3.5", "Q8"): "Qwen3.5-4B-Q8_needlebench.json",
    ("Qwen3.5", "Q4"): "Qwen3.5-4B-Q4_needlebench.json",
    ("Qwen3.5", "Q3"): "Qwen3.5-4B-Q3_needlebench.json",
    ("Qwen3.5", "UD-Q3"): "Qwen3.5-4B-UD-Q3_K_XL_needlebench.json",
    ("Qwen3", "Q8"): "Qwen3-4B-Q8_needlebench.json",
    ("Qwen3", "Q4"): "Qwen3-4B-Q4_needlebench.json",
    ("Qwen3", "Q3"): "Qwen3-4B-Q3_needlebench.json",
    ("Qwen3", "UD-Q3"): "Qwen3-4B-UD-Q3_K_XL_needlebench.json",
}

NOT_FOUND_MARKER = "NOT FOUND"


def classify_trial(trial):
    """Classify a trial as correct, refusal, or hallucination."""
    pred = trial.get("predicted", "")
    correct = trial.get("correct", False)
    if correct:
        return "correct"
    if NOT_FOUND_MARKER in str(pred):
        return "refusal"
    return "hallucination"


def analyze_refusals(data, task="M-RS"):
    """Analyze refusal rates for a model's needlebench data."""
    counts = defaultdict(int)
    total = 0

    for entry in data["tasks"][task]:
        for trial in entry.get("trials", []):
            label = classify_trial(trial)
            counts[label] += 1
            total += 1

    if total == 0:
        return None

    return {
        "total": total,
        "correct": counts["correct"],
        "refusal": counts["refusal"],
        "hallucination": counts["hallucination"],
        "refusal_pct": 100 * counts["refusal"] / total,
        "correct_pct": 100 * counts["correct"] / total,
        "hallucination_pct": 100 * counts["hallucination"] / total,
    }


def load_data():
    """Load all needlebench result files."""
    data = {}
    for key, fname in FILES.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"[!] Missing: {fname}")
            continue
        with open(path) as f:
            data[key] = json.load(f)
    return data


def print_table(data):
    """Print formatted refusal percentage table."""
    print("\n" + "=" * 70)
    print("REFUSAL PERCENTAGES BY MODEL AND QUANTIZATION TYPE")
    print("=" * 70)
    print()
    print(f"{'Model':<12} {'Quant':<8} {'Total':>8} {'Correct%':>10} {'Refusal%':>10} {'Halluc%':>10}")
    print("-" * 62)

    table_data = []

    for model in MODELS:
        for q in QUANTS:
            key = (model, q)
            if key not in data:
                row = f"{model:<12} {q:<8} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>10}"
                print(row)
                table_data.append((model, q, None, None, None, None))
                continue

            stats = analyze_refusals(data[key], "M-RS")
            if stats is None:
                row = f"{model:<12} {q:<8} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>10}"
                print(row)
                table_data.append((model, q, None, None, None, None))
                continue

            row = (
                f"{model:<12} {q:<8} "
                f"{stats['total']:>8} "
                f"{stats['correct_pct']:>9.1f}% "
                f"{stats['refusal_pct']:>9.1f}% "
                f"{stats['hallucination_pct']:>9.1f}%"
            )
            print(row)
            table_data.append((
                model, q,
                stats['total'],
                stats['correct_pct'],
                stats['refusal_pct'],
                stats['hallucination_pct']
            ))

    print()
    return table_data


def print_latex_table(table_data):
    """Print LaTeX-formatted table."""
    print("\n" + "=" * 70)
    print("LATEX TABLE FORMAT")
    print("=" * 70)
    print()
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{llrrrr}")
    print("\\toprule")
    print("Model & Quant & Total & Correct\\% & Refusal\\% & Halluc\\% \\\\")
    print("\\midrule")

    for model in MODELS:
        model_rows = [r for r in table_data if r[0] == model]
        for i, (m, q, total, correct, refusal, halluc) in enumerate(model_rows):
            if total is None:
                print(f"{m} & {q} & -- & -- & -- & -- \\\\")
            else:
                print(f"{m} & {q} & {total} & {correct:.1f} & {refusal:.1f} & {halluc:.1f} \\\\")
        if model != MODELS[-1]:
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Refusal Percentages by Model and Quantization Type}")
    print("\\label{tab:refusal-percentages}")
    print("\\end{table}")
    print()


def print_markdown_table(table_data):
    """Print Markdown-formatted table."""
    print("\n" + "=" * 70)
    print("MARKDOWN TABLE FORMAT")
    print("=" * 70)
    print()
    print("| Model | Quant | Total | Correct% | Refusal% | Halluc% |")
    print("|-------|-------|-------|----------|----------|---------|")

    for model in MODELS:
        model_rows = [r for r in table_data if r[0] == model]
        for m, q, total, correct, refusal, halluc in model_rows:
            if total is None:
                print(f"| {m} | {q} | -- | -- | -- | -- |")
            else:
                print(f"| {m} | {q} | {total} | {correct:.1f}% | {refusal:.1f}% | {halluc:.1f}% |")

    print()


def main():
    print("Loading needlebench results...")
    data = load_data()
    print(f"Loaded {len(data)} model configurations\n")

    table_data = print_table(data)
    print_latex_table(table_data)
    print_markdown_table(table_data)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model in MODELS:
        refusal_rates = []
        for q in QUANTS:
            key = (model, q)
            if key in data:
                stats = analyze_refusals(data[key], "M-RS")
                if stats:
                    refusal_rates.append((q, stats['refusal_pct']))

        if refusal_rates:
            print(f"\n{model}:")
            for q, rate in refusal_rates:
                print(f"  {q}: {rate:.1f}% refusal rate")
            avg = sum(r for _, r in refusal_rates) / len(refusal_rates)
            print(f"  Average: {avg:.1f}%")

    print()


if __name__ == "__main__":
    main()
