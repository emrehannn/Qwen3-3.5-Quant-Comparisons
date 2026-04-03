"""
Extract M-RS 8-K depth degradation graph from Figure 4 as a standalone PDF.

This script extracts only the M-RS (multi-fact reasoning) panel for the 8k context length
from the depth degradation analysis and saves it as a single-panel figure.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Directories ───────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/completed")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.3)
plt.rcParams["figure.dpi"] = 150

# Consistent color / marker scheme
COLORS  = {"Qwen3": "#4C72B0", "Qwen3.5": "#DD8452"}
MARKERS = {"Qwen3": "o",       "Qwen3.5": "s"}
QUANT_ORDER     = ["Q8", "Q4", "Q3", "Q3-UD"]
QUANT_LINESTYLE = {"Q8": "-",  "Q4": "--", "Q3": ":", "Q3-UD": "-."}

LABEL_MAP = {
    "Q8_0":      "Q8",
    "Q4_K_M":    "Q4",
    "Q3_K_M":    "Q3",
    "UD-Q3_K_XL": "Q3-UD",
}


def parse_model_config(model_name: str) -> tuple[str, str, str]:
    """Returns (arch_label, base_key, quant_label)."""
    if "Qwen3.5" in model_name:
        arch = "Qwen3.5 (GDN Hybrid)"
        base = "Qwen3.5"
    elif "Qwen3" in model_name:
        arch = "Qwen3 (Transformer)"
        base = "Qwen3"
    else:
        arch = base = model_name

    quant = "unknown"
    for pat in ["Q8_0", "Q4_K_M", "UD-Q3_K_XL", "Q3_K_M", "Q8", "Q4", "Q3"]:
        if pat in model_name:
            quant = LABEL_MAP.get(pat, pat)
            break

    return arch, base, quant


def load_all_results() -> dict:
    """Load all JSON result files from results/completed/."""
    results = defaultdict(dict)
    if not RESULTS_DIR.exists():
        print(f"[!] Results directory not found: {RESULTS_DIR}")
        return results

    known_benchmarks = ["needlebench", "perplexity", "gsm8k"]

    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        stem = json_file.stem
        found_bench = None
        model_slug = stem

        for bench in known_benchmarks:
            if stem.endswith(f"_{bench}"):
                found_bench = bench
                model_slug = stem[: -len(f"_{bench}")]
                break

        if found_bench is None:
            continue

        try:
            with open(json_file) as fh:
                data = json.load(fh)
            results[found_bench][model_slug] = data
        except Exception as exc:
            print(f"[!] Error loading {json_file}: {exc}")

    return results


def extract_mrs_8k_depth_graph(results: dict, output_name: str = "figure4_mrs_8k_only") -> None:
    """
    Extract and save only the M-RS 8-K depth degradation panel as a standalone figure.
    
    Args:
        results: Loaded benchmark results dict
        output_name: Base filename for the output (without extension)
    """
    bench_key = "needlebench"
    if bench_key not in results or not results[bench_key]:
        print("[skip] No NeedleBench data available")
        return

    TARGET_CTX = 8192  # 8k context length
    TARGET_TASK = "M-RS"

    # curves[base][quant] = {depth_pct: score%}
    curves = defaultdict(lambda: defaultdict(dict))

    for model_name, result in results[bench_key].items():
        _, base, quant = parse_model_config(model_name)

        # M-RS uses avg_score for continuous scoring
        metric = "avg_score"
        task_results = result.get("tasks", {}).get(TARGET_TASK, [])
        
        for task_result in task_results:
            ctx = task_result.get("context_length", 0)
            breakdown = task_result.get("depth_breakdown", {})
            if ctx != TARGET_CTX or not breakdown:
                continue
            
            for depth_label, stats in breakdown.items():
                try:
                    depth_pct = int(depth_label.rstrip("%"))
                except ValueError:
                    continue
                val = stats.get(metric, None)
                if val is not None:
                    curves[base][quant][depth_pct] = val * 100

    if not curves:
        print(f"[skip] No depth_breakdown data for {TARGET_TASK} at {TARGET_CTX//1024}k context")
        return

    # Create single panel figure with more height for legend space
    fig, ax = plt.subplots(figsize=(10, 7))

    for base in sorted(curves):
        for quant in QUANT_ORDER:
            depth_acc = curves[base].get(quant, {})
            if not depth_acc:
                continue
            depths = sorted(depth_acc)
            values = [depth_acc[d] for d in depths]
            ax.plot(
                depths, values,
                color=COLORS.get(base, "gray"),
                linestyle=QUANT_LINESTYLE.get(quant, "-"),
                marker=MARKERS.get(base, "o"),
                linewidth=2.5, markersize=9,
                label=f"{base} {quant}",
            )

    ax.set_xlabel("Needle depth (%)")
    ax.set_ylabel("M-RS Avg Score (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([5, 10, 30, 50, 70, 90])
    ax.grid(True, alpha=0.3)

    # Unified legend using proxy artists
    arch_handles = [
        mlines.Line2D([], [], color=COLORS[b], marker=MARKERS[b],
                      linewidth=2.5, markersize=10, label=b)
        for b in ["Qwen3", "Qwen3.5"] if b in COLORS
    ]
    quant_handles = [
        mlines.Line2D([], [], color="gray", linestyle=QUANT_LINESTYLE[q],
                      linewidth=2.5, label=q)
        for q in QUANT_ORDER
    ]
    ax.legend(handles=arch_handles + quant_handles,
              loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, -0.3), frameon=True)

    ax.set_title("M-RS 8k — Multi-Fact Reasoning Depth Degradation",
                 fontweight="bold", fontsize=14)

    plt.tight_layout()
    
    # Save as PDF only
    pdf_path = FIGURES_DIR / f"{output_name}.pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {pdf_path}")
    plt.close(fig)


def main():
    print("Loading benchmark results...")
    results = load_all_results()

    if "needlebench" not in results or not results["needlebench"]:
        print("No NeedleBench results found. Run benchmarks first.")
        return

    print("\nExtracting M-RS 8-K depth degradation graph...")
    extract_mrs_8k_depth_graph(results)

    print(f"\nDone. Figure saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
