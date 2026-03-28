"""
Analysis script for benchmark results.
Generates publication-quality figures for the quantization study.

All NeedleBench metrics in result files are stored as fractions in [0, 1].
This script multiplies by 100 when displaying percentages.

Figures:
    1. Perplexity degradation curve
    2. GSM8K accuracy (short-context control)
    3. NeedleBench heatmap — task × context length × quant
    4. Depth-degradation curves — THE THESIS FIGURE
       Two-row layout: top = S-RT, bottom = M-RS.
       Shows per-depth accuracy for each (arch, quant) combination.
       If GDN compounds quantization error over sequence distance, Qwen3.5
       should show steeper accuracy decline at shallow depths vs Qwen3.
       M-RS (multi-fact reasoning) is the PRIMARY hypothesis task.
    5. Quantization degradation delta — Q8 baseline vs Q3 stress test
    6. M-RT accuracy depth curves — per-depth accuracy for multi-needle retrieval
    7. Per-task Q8→Q3 degradation deltas — breakout by S-RT, M-RT, M-RS
"""

import json
import matplotlib
matplotlib.use("Agg")   # headless — safe for servers without a display
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
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.dpi"] = 150

# Consistent color / marker scheme
COLORS  = {"Qwen3": "#1f77b4", "Qwen3.5": "#ff7f0e"}
MARKERS = {"Qwen3": "o",       "Qwen3.5": "s"}
QUANT_ORDER     = ["Q8", "Q4", "Q3"]
QUANT_LINESTYLE = {"Q8": "-",  "Q4": "--", "Q3": ":"}

LABEL_MAP = {
    "Q8_0":      "Q8",
    "Q4_K_M":    "Q4",
    "Q3_K_M":    "Q3",
    "UD-Q3_K_XL": "Q3",
}


# ══════════════════════════════════════════════════════════════════════════════
# Loading & parsing
# ══════════════════════════════════════════════════════════════════════════════

def load_all_results() -> dict:
    """
    Load all JSON result files from results/completed/.
    Returns nested dict: results[benchmark][model_slug] = data_dict
    """
    results: dict = defaultdict(dict)
    if not RESULTS_DIR.exists():
        print(f"[!] Results directory not found: {RESULTS_DIR}")
        return results

    known_benchmarks = ["needlebench", "perplexity", "gsm8k"]

    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        stem      = json_file.stem
        found_bench = None
        model_slug  = stem

        for bench in known_benchmarks:
            if stem.endswith(f"_{bench}"):
                found_bench = bench
                model_slug  = stem[: -len(f"_{bench}")]
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


def parse_model_config(model_name: str) -> tuple[str, str, str]:
    """
    Returns (arch_label, base_key, quant_label).
    base_key is "Qwen3" or "Qwen3.5" — used as dict keys / legend labels.
    """
    if "Qwen3.5" in model_name:
        arch  = "Qwen3.5 (GDN Hybrid)"
        base  = "Qwen3.5"
    elif "Qwen3" in model_name:
        arch  = "Qwen3 (Transformer)"
        base  = "Qwen3"
    else:
        arch = base = model_name

    quant = "unknown"
    for pat in ["Q8_0", "Q4_K_M", "UD-Q3_K_XL", "Q3_K_M", "Q8", "Q4", "Q3"]:
        if pat in model_name:
            quant = LABEL_MAP.get(pat, pat)
            break

    return arch, base, quant


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Perplexity
# ══════════════════════════════════════════════════════════════════════════════

def plot_perplexity(results: dict) -> None:
    bench = "perplexity"
    if bench not in results or not results[bench]:
        print("[skip] Figure 1: no perplexity data")
        return

    ppl_data: dict = defaultdict(dict)
    for model_name, result in results[bench].items():
        _, base, quant = parse_model_config(model_name)
        ppl = result.get("perplexity", float("nan"))
        ppl_data[base][quant] = ppl

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(QUANT_ORDER))

    for model in sorted(ppl_data):
        values = [ppl_data[model].get(q, np.nan) for q in QUANT_ORDER]
        ax.plot(x_pos, values,
                marker=MARKERS.get(model, "o"), linewidth=2.5, markersize=10,
                label=model, color=COLORS.get(model, "gray"))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(QUANT_ORDER)
    ax.set_xlabel("Quantisation Level")
    ax.set_ylabel("Perplexity (lower = better)")
    ax.set_title("Figure 1: WikiText-103 Perplexity by Quantisation Level",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, "figure1_perplexity")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — GSM8K
# ══════════════════════════════════════════════════════════════════════════════

def plot_gsm8k(results: dict) -> None:
    bench = "gsm8k"
    if bench not in results or not results[bench]:
        print("[skip] Figure 2: no GSM8K data")
        return

    acc_data: dict = defaultdict(dict)
    for model_name, result in results[bench].items():
        _, base, quant = parse_model_config(model_name)
        acc_data[base][quant] = result.get("accuracy", 0.0) * 100   # 0-1 → %

    fig, ax = plt.subplots(figsize=(8, 5))
    x     = np.arange(len(QUANT_ORDER))
    width = 0.35
    models = sorted(acc_data)

    for i, model in enumerate(models):
        values = [acc_data[model].get(q, 0.0) for q in QUANT_ORDER]
        offset = width * (i - (len(models) - 1) / 2)
        ax.bar(x + offset, values, width,
               label=model, color=COLORS.get(model, "gray"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(QUANT_ORDER)
    ax.set_xlabel("Quantisation Level")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Figure 2: GSM8K Accuracy — Short-Context Control (250 samples)",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "figure2_gsm8k")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — NeedleBench heatmap (context length × quant per task per arch)
# ══════════════════════════════════════════════════════════════════════════════

def plot_needlebench_heatmap(results: dict) -> None:
    """
    Heatmap: rows = context lengths, columns = quant levels.
    One subplot per (architecture × task) combination.
    Scores are multiplied by 100 for percentage display.

    NOTE: S-RT and M-RS store 'accuracy' in [0,1]; M-RT stores 'f1' in [0,1].
    Both are multiplied by 100 here.
    """
    bench_key = "needlebench"
    if bench_key not in results or not results[bench_key]:
        print("[skip] Figure 3: no NeedleBench data")
        return

    task_metric = {"S-RT": "accuracy", "M-RT": "accuracy", "M-RS": "accuracy"}
    tasks       = ["S-RT", "M-RT", "M-RS"]

    # Collect data: models_data[base][task][ctx_len][quant] = score (%)
    models_data    = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    all_ctx_lengths = set()

    for model_name, result in results[bench_key].items():
        _, base, quant = parse_model_config(model_name)
        for task in tasks:
            metric       = task_metric[task]
            task_results = result.get("tasks", {}).get(task, [])
            for r in task_results:
                ctx = r.get("context_length", 0)
                if ctx == 0:
                    continue
                all_ctx_lengths.add(ctx)
                val = r.get(metric, None)
                if val is not None:
                    # Both accuracy and f1 are in [0,1] — multiply to get %
                    models_data[base][task][ctx][quant] = val * 100

    CTX_LENGTHS  = sorted(all_ctx_lengths)
    model_bases  = sorted(models_data.keys())
    if not model_bases or not CTX_LENGTHS:
        print("[skip] Figure 3: insufficient data to build heatmap")
        return

    fig, axes = plt.subplots(
        len(model_bases), len(tasks),
        figsize=(5 * len(tasks), 4 * len(model_bases))
    )
    if len(model_bases) == 1:
        axes = axes.reshape(1, -1)
    if len(tasks) == 1:
        axes = axes.reshape(-1, 1)

    for i, model in enumerate(model_bases):
        for j, task in enumerate(tasks):
            ax     = axes[i, j]
            matrix = np.full((len(CTX_LENGTHS), len(QUANT_ORDER)), np.nan)
            annot  = []

            for ki, ctx in enumerate(CTX_LENGTHS):
                row = []
                for kj, quant in enumerate(QUANT_ORDER):
                    val = models_data[model][task].get(ctx, {}).get(quant, None)
                    if val is not None:
                        matrix[ki, kj] = val
                        row.append(f"{val:.0f}%")
                    else:
                        row.append("N/A")
                annot.append(row)

            sns.heatmap(matrix, ax=ax, annot=annot, fmt="",
                        cmap="RdYlGn", vmin=0, vmax=100,
                        linewidths=0.5, linecolor="white",
                        xticklabels=QUANT_ORDER,
                        yticklabels=[f"{c//1024}k" for c in CTX_LENGTHS],
                        cbar_kws={"label": "Score (%)"} if j == len(tasks)-1
                                  else {"label": ""},
                        mask=np.isnan(matrix))

            ax.set_title(f"{model} — {task} (Accuracy)",
                         fontsize=10, fontweight="bold")
            if i == len(model_bases) - 1:
                ax.set_xlabel("Quantisation")
            if j == 0:
                ax.set_ylabel("Context Length")

    fig.suptitle(
        "Figure 3: NeedleBench Performance by Task, Context Length & Quantisation",
        fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "figure3_needlebench_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Depth-degradation (THESIS FIGURE)
# ══════════════════════════════════════════════════════════════════════════════

def plot_depth_degradation(results: dict) -> None:
    """
    THE THESIS FIGURE.

    Two-row layout:
      Top row:    S-RT accuracy (%) vs needle depth — single retrieval
      Bottom row: M-RS avg_score (%) vs needle depth — multi-fact reasoning
                  (continuous score instead of binary accuracy for smoother curves)

    One column per context length.  M-RS is the PRIMARY hypothesis task:
    if GDN compounds quantization error, the multi-fact reasoning signal
    should degrade more steeply than single retrieval.

    Scores are multiplied by 100 for percentage display.
    """
    bench_key = "needlebench"
    if bench_key not in results or not results[bench_key]:
        print("[skip] Figure 4: no NeedleBench data")
        return

    # ── Collect depth curves for both S-RT and M-RS ──────────────────────
    # curves[task][ctx_len][base][quant] = {depth_pct: score%}
    curves: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    all_ctx_lengths: set = set()

    for model_name, result in results[bench_key].items():
        _, base, quant = parse_model_config(model_name)

        for task_key in ("S-RT", "M-RS"):
            # S-RT: binary accuracy (high baseline, clear signal)
            # M-RS: avg_score (continuous, smoother curves at low baseline)
            metric = "avg_score" if task_key == "M-RS" else "accuracy"
            task_results = result.get("tasks", {}).get(task_key, [])
            for task_result in task_results:
                ctx       = task_result.get("context_length", 0)
                breakdown = task_result.get("depth_breakdown", {})
                if not ctx or not breakdown:
                    continue
                all_ctx_lengths.add(ctx)
                for depth_label, stats in breakdown.items():
                    try:
                        depth_pct = int(depth_label.rstrip("%"))
                    except ValueError:
                        continue
                    val = stats.get(metric, None)
                    if val is not None:
                        curves[task_key][ctx][base][quant][depth_pct] = val * 100

    # Determine which tasks actually have data
    tasks_with_data = [t for t in ("S-RT", "M-RS") if curves.get(t)]
    if not tasks_with_data:
        print("[skip] Figure 4: no depth_breakdown data in S-RT or M-RS results")
        return

    ctx_lengths = sorted(all_ctx_lengths)
    n_cols      = len(ctx_lengths)
    n_rows      = len(tasks_with_data)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 5 * n_rows),
                             sharey=True, squeeze=False)

    for row_idx, task_key in enumerate(tasks_with_data):
        for col_idx, ctx in enumerate(ctx_lengths):
            ax       = axes[row_idx, col_idx]
            ctx_data = curves[task_key].get(ctx, {})
            if not ctx_data:
                ax.set_title(f"{task_key} — {ctx // 1024}k — no data")
                continue

            for base in sorted(ctx_data):
                for quant in QUANT_ORDER:
                    depth_acc = ctx_data[base].get(quant, {})
                    if not depth_acc:
                        continue
                    depths = sorted(depth_acc)
                    values = [depth_acc[d] for d in depths]
                    ax.plot(
                        depths, values,
                        color=COLORS.get(base, "gray"),
                        linestyle=QUANT_LINESTYLE.get(quant, "-"),
                        marker=MARKERS.get(base, "o"),
                        linewidth=2, markersize=7,
                        label=f"{base} {quant}",
                    )

            ax.set_title(f"{task_key} — {ctx // 1024}k", fontweight="bold")
            ax.set_xlabel("Needle depth (%)\n(5% = near start, 90% = near end)")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xticks([5, 10, 30, 50, 70, 90])
            ax.grid(True, alpha=0.3)

        # y-axis label on leftmost panel of each row
        y_label = "M-RS Avg Score (%)" if task_key == "M-RS" else f"{task_key} Accuracy (%)"
        axes[row_idx, 0].set_ylabel(y_label)

    # Unified legend using proxy artists
    arch_handles = [
        mlines.Line2D([], [], color=COLORS[b], marker=MARKERS[b],
                      linewidth=2, markersize=8, label=b)
        for b in ["Qwen3", "Qwen3.5"] if b in COLORS
    ]
    quant_handles = [
        mlines.Line2D([], [], color="gray", linestyle=QUANT_LINESTYLE[q],
                      linewidth=2, label=q)
        for q in QUANT_ORDER
    ]
    fig.legend(handles=arch_handles + quant_handles,
               loc="lower center", ncol=len(arch_handles) + len(quant_handles),
               bbox_to_anchor=(0.5, -0.06), frameon=True)

    fig.suptitle(
        "Figure 4: Depth-Degradation — S-RT (single retrieval) vs M-RS (multi-fact reasoning)\n"
        "Hypothesis: GDN (Qwen3.5) compounds quantization error at shallow depths,\n"
        "especially under multi-fact reasoning pressure",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "figure4_depth_degradation")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Degradation delta (Q8 → Q3)
# ══════════════════════════════════════════════════════════════════════════════

def plot_deltas(results: dict) -> None:
    """
    Bar chart showing accuracy/score drop from Q8 to Q3 per benchmark.
    Benchmarks: GSM8K (short-context control) and NeedleBench overall.

    A larger delta for Qwen3.5 vs Qwen3 on NeedleBench—but not GSM8K—
    supports the architecture-specific long-context degradation hypothesis.
    """
    benchmarks_to_plot = []

    # --- GSM8K ---
    if results.get("gsm8k"):
        per_model: dict = defaultdict(dict)
        for model_name, result in results["gsm8k"].items():
            _, base, quant = parse_model_config(model_name)
            per_model[base][quant] = result.get("accuracy", 0.0) * 100  # [0,1]→%

        deltas = {m: vals.get("Q8", 0) - vals.get("Q3", 0)
                  for m, vals in per_model.items()
                  if "Q8" in vals and "Q3" in vals}
        if deltas:
            benchmarks_to_plot.append(("GSM8K\n(Q8→Q3 drop)", deltas))

    # --- NeedleBench overall ---
    if results.get("needlebench"):
        # Build a fresh per_model dict — do NOT reuse from GSM8K block
        nb_per_model: dict = defaultdict(dict)
        for model_name, result in results["needlebench"].items():
            _, base, quant = parse_model_config(model_name)
            # composite_scores["Overall"] is in [0,1]
            score = result.get("composite_scores", {}).get("Overall", None)
            if score is not None:
                nb_per_model[base][quant] = score * 100   # → %

        deltas = {m: vals.get("Q8", 0) - vals.get("Q3", 0)
                  for m, vals in nb_per_model.items()
                  if "Q8" in vals and "Q3" in vals}
        if deltas:
            benchmarks_to_plot.append(("NeedleBench\n(Q8→Q3 drop)", deltas))

    if not benchmarks_to_plot:
        print("[skip] Figure 5: insufficient Q8+Q3 pairs for delta chart")
        return

    bench_names = [b[0] for b in benchmarks_to_plot]
    x     = np.arange(len(bench_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model in enumerate(["Qwen3", "Qwen3.5"]):
        values = [bench_data.get(model, 0) for _, bench_data in benchmarks_to_plot]
        offset = width * (i - 0.5)
        bars   = ax.bar(x + offset, values, width,
                        label=model, color=COLORS.get(model, "gray"), alpha=0.85)
        for bar, val in zip(bars, values):
            if val != 0:
                ax.annotate(f"{val:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(bench_names, fontsize=11)
    ax.set_ylabel("Accuracy Drop (percentage points)")
    ax.set_title("Figure 5: Quantisation Degradation Q8 → Q3\n"
                 "A larger NeedleBench drop for Qwen3.5 supports the GDN hypothesis",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "figure5_deltas")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — M-RT Accuracy Depth Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_mrt_depth(results: dict) -> None:
    """
    M-RT accuracy depth curves.

    Plots M-RT accuracy (%) vs needle depth for each (architecture, quant)
    combination, one panel per context length.

    Shows whether multi-needle retrieval degrades at specific depths,
    completing the depth profile alongside S-RT and M-RS.
    """
    bench_key = "needlebench"
    if bench_key not in results or not results[bench_key]:
        print("[skip] Figure 6: no NeedleBench data")
        return

    # curves[ctx_len][base][quant] = {depth_pct: accuracy%}
    curves: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    all_ctx_lengths: set = set()

    for model_name, result in results[bench_key].items():
        _, base, quant = parse_model_config(model_name)

        mrt_results = result.get("tasks", {}).get("M-RT", [])
        for task_result in mrt_results:
            ctx       = task_result.get("context_length", 0)
            breakdown = task_result.get("depth_breakdown", {})
            if not ctx or not breakdown:
                continue
            all_ctx_lengths.add(ctx)
            for depth_label, stats in breakdown.items():
                try:
                    depth_pct = int(depth_label.rstrip("%"))
                except ValueError:
                    continue
                acc = stats.get("accuracy", None)
                if acc is not None:
                    curves[ctx][base][quant][depth_pct] = acc * 100

    if not curves:
        print("[skip] Figure 6: no depth_breakdown data in M-RT results")
        return

    ctx_lengths = sorted(all_ctx_lengths)
    n_panels    = len(ctx_lengths)

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, ctx in zip(axes, ctx_lengths):
        ctx_data = curves[ctx]
        if not ctx_data:
            ax.set_title(f"{ctx // 1024}k — no data")
            continue

        for base in sorted(ctx_data):
            for quant in QUANT_ORDER:
                depth_acc = ctx_data[base].get(quant, {})
                if not depth_acc:
                    continue
                depths = sorted(depth_acc)
                values = [depth_acc[d] for d in depths]
                ax.plot(
                    depths, values,
                    color=COLORS.get(base, "gray"),
                    linestyle=QUANT_LINESTYLE.get(quant, "-"),
                    marker=MARKERS.get(base, "o"),
                    linewidth=2, markersize=7,
                    label=f"{base} {quant}",
                )

        ax.set_title(f"Context length: {ctx // 1024}k", fontweight="bold")
        ax.set_xlabel("Needle depth (%)\n(5% = near start, 90% = near end)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks([5, 10, 30, 50, 70, 90])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("M-RT Accuracy (%)")

    arch_handles = [
        mlines.Line2D([], [], color=COLORS[b], marker=MARKERS[b],
                      linewidth=2, markersize=8, label=b)
        for b in ["Qwen3", "Qwen3.5"] if b in COLORS
    ]
    quant_handles = [
        mlines.Line2D([], [], color="gray", linestyle=QUANT_LINESTYLE[q],
                      linewidth=2, label=q)
        for q in QUANT_ORDER
    ]
    fig.legend(handles=arch_handles + quant_handles,
               loc="lower center", ncol=len(arch_handles) + len(quant_handles),
               bbox_to_anchor=(0.5, -0.08), frameon=True)

    fig.suptitle(
        "Figure 6: M-RT Accuracy by Needle Depth — Multi-Needle Retrieval",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "figure6_mrt_depth")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Per-task Q8→Q3 degradation deltas
# ══════════════════════════════════════════════════════════════════════════════

def plot_per_task_deltas(results: dict) -> None:
    """
    Bar chart of Q8→Q3 accuracy/F1 drop broken out by individual task.

    Groups on x-axis: GSM8K, S-RT, M-RT, M-RS.
    Two bars per group (Qwen3 vs Qwen3.5).

    This directly answers which task type is most affected by quantization
    for each architecture — the heart of the thesis.
    """
    benchmarks_to_plot = []

    # --- GSM8K ---
    if results.get("gsm8k"):
        per_model: dict = defaultdict(dict)
        for model_name, result in results["gsm8k"].items():
            _, base, quant = parse_model_config(model_name)
            per_model[base][quant] = result.get("accuracy", 0.0) * 100

        deltas = {m: vals.get("Q8", 0) - vals.get("Q3", 0)
                  for m, vals in per_model.items()
                  if "Q8" in vals and "Q3" in vals}
        if deltas:
            benchmarks_to_plot.append(("GSM8K", deltas))

    # --- NeedleBench per-task (all tasks now use accuracy) ---
    if results.get("needlebench"):
        for task_key in ["S-RT", "M-RT", "M-RS"]:
            nb_per_model: dict = defaultdict(dict)
            for model_name, result in results["needlebench"].items():
                _, base, quant = parse_model_config(model_name)
                task_results = result.get("tasks", {}).get(task_key, [])
                if not task_results:
                    continue
                # Average accuracy across context lengths
                vals = [r.get("accuracy", 0.0) for r in task_results]
                avg  = (sum(vals) / len(vals)) * 100 if vals else 0.0
                nb_per_model[base][quant] = avg

            deltas = {m: vals.get("Q8", 0) - vals.get("Q3", 0)
                      for m, vals in nb_per_model.items()
                      if "Q8" in vals and "Q3" in vals}
            if deltas:
                benchmarks_to_plot.append((task_key, deltas))

    if not benchmarks_to_plot:
        print("[skip] Figure 7: insufficient Q8+Q3 pairs for per-task delta chart")
        return

    bench_names = [b[0] for b in benchmarks_to_plot]
    x     = np.arange(len(bench_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model in enumerate(["Qwen3", "Qwen3.5"]):
        values = [bench_data.get(model, 0) for _, bench_data in benchmarks_to_plot]
        offset = width * (i - 0.5)
        bars   = ax.bar(x + offset, values, width,
                        label=model, color=COLORS.get(model, "gray"), alpha=0.85)
        for bar, val in zip(bars, values):
            if val != 0:
                ax.annotate(f"{val:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(bench_names, fontsize=11)
    ax.set_ylabel("Score Drop (percentage points, Q8 → Q3)")
    ax.set_title("Figure 7: Per-Task Quantisation Degradation Q8 → Q3\n"
                 "Which task type suffers most from quantization in each architecture?",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "figure7_per_task_deltas")


# ══════════════════════════════════════════════════════════════════════════════
# Console summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(results: dict) -> None:
    sep = "=" * 80
    print(f"\n{sep}\nBENCHMARK SUMMARY\n{sep}")

    if results.get("perplexity"):
        print("\nWikiText-103 Perplexity (lower = better):")
        print("-" * 60)
        for name in sorted(results["perplexity"]):
            ppl = results["perplexity"][name].get("perplexity", float("nan"))
            print(f"  {name:40s}: {ppl:.2f}")

    if results.get("gsm8k"):
        print("\nGSM8K Accuracy — 250 samples (higher = better):")
        print("-" * 60)
        for name in sorted(results["gsm8k"]):
            r   = results["gsm8k"][name]
            acc = r.get("accuracy", 0.0) * 100  # [0,1] → %
            c, t = r.get("correct", 0), r.get("total", 0)
            print(f"  {name:40s}: {acc:.1f}%  ({c}/{t})")

    if results.get("needlebench"):
        print("\nNeedleBench Results (higher = better; all scores × 100 for display):")
        print("-" * 60)
        for name in sorted(results["needlebench"]):
            result    = results["needlebench"][name]
            composite = result.get("composite_scores", {})

            # composite_scores stores [0,1] values — multiply by 100 for display
            overall = composite.get("Overall", 0.0) * 100
            s_rt    = composite.get("S-RT",    0.0) * 100
            m_rt    = composite.get("M-RT",    0.0) * 100
            m_rs    = composite.get("M-RS",    0.0) * 100

            print(f"\n  {name}")
            print(f"    Overall : {overall:.2f}%  "
                  f"(S-RT {s_rt:.1f}% | M-RT {m_rt:.1f}% | M-RS {m_rs:.1f}%)")

            for task in ["S-RT", "M-RT", "M-RS"]:
                task_results = result.get("tasks", {}).get(task, [])
                if not task_results:
                    continue
                print(f"    {task}:")
                for r in task_results:
                    ctx   = r.get("context_length", 0)
                    score = r.get("accuracy", 0.0) * 100   # [0,1] → %
                    print(f"      {ctx//1024}k: {score:.1f}%")

                    # Print depth breakdown if present
                    bd = r.get("depth_breakdown", {})
                    if bd:
                        for dlabel, dstats in sorted(
                            bd.items(), key=lambda x: int(x[0].rstrip("%"))
                        ):
                            dscore = dstats.get("accuracy", 0.0) * 100
                            print(f"        depth {dlabel}: {dscore:.1f}%")

    print(f"\n{sep}")


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {FIGURES_DIR / stem}.{{png,pdf}}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Loading benchmark results...")
    results = load_all_results()

    if not any(results.values()):
        print("No results found in results/completed/. Run benchmarks first.")
        return

    print_summary_table(results)

    print("\nGenerating figures...")
    plot_perplexity(results)          # Figure 1
    plot_gsm8k(results)               # Figure 2
    plot_needlebench_heatmap(results) # Figure 3
    plot_depth_degradation(results)   # Figure 4 — THESIS FIGURE (S-RT + M-RS)
    plot_deltas(results)              # Figure 5
    plot_mrt_depth(results)           # Figure 6 — M-RT F1 depth curves
    plot_per_task_deltas(results)     # Figure 7 — per-task Q8→Q3 deltas

    print(f"\nAll figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()