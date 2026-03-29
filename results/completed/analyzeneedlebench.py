"""
NeedleBench Diagnostic Analysis
Qwen3 (Transformer) vs Qwen3.5 (GDN) — Quantization Robustness Study
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# ── 1. Load all files ──────────────────────────────────────────────────────────

FILES = {
    ("Qwen3.5", "Q8"): "Qwen3.5-4B-Q8_needlebench.json",
    ("Qwen3.5", "Q4"): "Qwen3.5-4B-Q4_needlebench.json",
    ("Qwen3.5", "Q3"): "Qwen3.5-4B-Q3_needlebench.json",
    ("Qwen3.5", "UD-Q3"): "Qwen3.5-4B-UD-Q3_K_XL_needlebench.json",
    ("Qwen3",   "Q8"): "Qwen3-4B-Q8_needlebench.json",
    ("Qwen3",   "Q4"): "Qwen3-4B-Q4_needlebench.json",
    ("Qwen3",   "Q3"): "Qwen3-4B-Q3_needlebench.json",
    ("Qwen3",   "UD-Q3"): "Qwen3-4B-UD-Q3_K_XL_needlebench.json",
}

BASE = Path(__file__).parent
data = {}
for key, fname in FILES.items():
    path = BASE / fname
    if not path.exists():
        print(f"Skipping {key} (file {fname} not found yet)")
        continue
    with open(path) as f:
        data[key] = json.load(f)
    print(f"Loaded {key}: {data[key].get('model', 'Unknown')}")

print()

# ── 2. Build flat summary tables ───────────────────────────────────────────────

def get_accuracy(d, task, ctx):
    for entry in d["tasks"][task]:
        if entry["context_length"] == ctx:
            return round(entry["accuracy"] * 100, 2)
    return None

TASKS   = ["S-RT", "M-RT", "M-RS"]
CTXS    = [4096, 8192, 16384]
MODELS  = ["Qwen3", "Qwen3.5"]
QUANTS  = ["Q8", "Q4", "Q3", "UD-Q3"]
CTX_LABELS = {4096: "4k", 8192: "8k", 16384: "16k"}

print("=" * 70)
print("TABLE 1 — Accuracy by Task / Context / Quantization")
print("=" * 70)

for task in TASKS:
    print(f"\n── {task} ──────────────────────────────────────────")
    header = f"{'Model':<10} {'Quant':<6}" + "".join(f"  {CTX_LABELS[c]:>5}" for c in CTXS) + "  Δ(4k→16k)"
    print(header)
    print("-" * len(header))
    for model in MODELS:
        for q in QUANTS:
            key = (model, q)
            if key not in data:
                continue
            vals = [get_accuracy(data[key], task, c) for c in CTXS]
            delta = None if (vals[0] is None or vals[-1] is None) else round(vals[-1] - vals[0], 1)
            row = f"{model:<10} {q:<6}"
            for v in vals:
                row += f"  {v:>5.1f}" if v is not None else f"  {'N/A':>5}"
            row += f"  {delta:>+.1f}pp" if delta is not None else "    N/A"
            print(row)

# ── 3. Degradation slope analysis ─────────────────────────────────────────────

print("\n")
print("=" * 70)
print("TABLE 2 — Context-Degradation Slope (Δ 4k→16k pp)")
print("Positive = improvement, Negative = degradation")
print("=" * 70)

for task in TASKS:
    print(f"\n{task}:")
    print(f"  {'Quant':<6}  {'Qwen3 Δ':>10}  {'Qwen3.5 Δ':>12}  {'Diff (GDN - Transformer)':>26}")
    print("  " + "-" * 62)
    for q in QUANTS:
        qwen3_key   = ("Qwen3",   q)
        qwen35_key  = ("Qwen3.5", q)
        if qwen3_key not in data or qwen35_key not in data:
            continue
        d3   = [get_accuracy(data[qwen3_key],  task, c) for c in CTXS]
        d35  = [get_accuracy(data[qwen35_key], task, c) for c in CTXS]
        delta3  = round(d3[-1]  - d3[0],  1) if None not in d3  else None
        delta35 = round(d35[-1] - d35[0], 1) if None not in d35 else None
        diff = round(delta35 - delta3, 1) if (delta3 is not None and delta35 is not None) else None
        note = " ← GDN MORE ROBUST" if (diff is not None and diff > 5) else ""
        note = " ← TRANSFORMER MORE ROBUST" if (diff is not None and diff < -5) else note
        print(f"  {q:<6}  {delta3:>+9.1f}pp  {delta35:>+11.1f}pp  {diff:>+25.1f}pp{note}")

# ── 4. Q4 Anomaly: Q4 vs Q3 for Qwen3 ────────────────────────────────────────

print("\n")
print("=" * 70)
print("TABLE 3 — Q4 Anomaly Analysis: Qwen3 Q4 vs Q3 vs Q8")
print("(Is Q4 worse than the more-aggressively-quantized Q3?)")
print("=" * 70)

for task in TASKS:
    print(f"\n{task}:")
    print(f"  {'Ctx':<6}  {'Q8':>6}  {'Q4':>6}  {'Q3':>6}  {'Q4 worse than Q3?':>20}")
    print("  " + "-" * 48)
    for ctx in CTXS:
        q8  = get_accuracy(data[("Qwen3", "Q8")],  task, ctx)
        q4  = get_accuracy(data[("Qwen3", "Q4")],  task, ctx)
        q3  = get_accuracy(data[("Qwen3", "Q3")],  task, ctx)
        anomaly = "YES ⚠️" if (q4 is not None and q3 is not None and q4 < q3) else "no"
        print(f"  {CTX_LABELS[ctx]:<6}  {q8:>6.1f}  {q4:>6.1f}  {q3:>6.1f}  {anomaly:>20}")

# ── 5. Refusal vs Hallucination analysis (M-RS trials) ────────────────────────

print("\n")
print("=" * 70)
print("TABLE 4 — Failure Mode Analysis: M-RS Refusal vs Hallucination")
print("(Based on trial-level predicted text)")
print("=" * 70)

NOT_FOUND_MARKER = "NOT FOUND"

def classify_trial(trial):
    """Classify a failed trial as refusal or hallucination."""
    pred = trial.get("predicted", "")
    correct = trial.get("correct", False)
    if correct:
        return "correct"
    if NOT_FOUND_MARKER in str(pred):
        return "refusal"
    return "hallucination"

def analyze_failures(d, task="M-RS"):
    counts = defaultdict(int)
    ctx_breakdown = defaultdict(lambda: defaultdict(int))
    for entry in d["tasks"][task]:
        ctx = entry["context_length"]
        for trial in entry.get("trials", []):
            label = classify_trial(trial)
            counts[label] += 1
            ctx_breakdown[ctx][label] += 1
    return counts, ctx_breakdown

print(f"\n{'Model':<10} {'Quant':<6} {'Total':>7} {'Correct%':>10} {'Refusal%':>10} {'Halluc%':>10}")
print("-" * 58)

refusal_data = {}
for model in MODELS:
    for q in QUANTS:
        key = (model, q)
        if key not in data:
            continue
        counts, ctx_bd = analyze_failures(data[key], "M-RS")
        total = sum(counts.values())
        correct_pct  = 100 * counts["correct"]      / total if total else 0
        refusal_pct  = 100 * counts["refusal"]      / total if total else 0
        halluc_pct   = 100 * counts["hallucination"] / total if total else 0
        refusal_data[key] = {"total": total, "correct_pct": correct_pct,
                              "refusal_pct": refusal_pct, "halluc_pct": halluc_pct,
                              "ctx_breakdown": ctx_bd}
        print(f"{model:<10} {q:<6} {total:>7} {correct_pct:>9.1f}% {refusal_pct:>9.1f}% {halluc_pct:>9.1f}%")

# ── 6. Refusal rate by context length ─────────────────────────────────────────

print("\n")
print("TABLE 5 — M-RS Refusal Rate by Context Length")
print("(Does refusal rate increase with context = model admitting context overload?)")
print()

for model in MODELS:
    print(f"  {model}:")
    print(f"    {'Quant':<6}" + "".join(f"  {CTX_LABELS[c]:>6} ref%" for c in CTXS))
    print("    " + "-" * 42)
    for q in QUANTS:
        key = (model, q)
        if key not in refusal_data:
            continue
        ctx_bd = refusal_data[key]["ctx_breakdown"]
        row = f"    {q:<6}"
        for ctx in CTXS:
            bd = ctx_bd[ctx]
            total_ctx = sum(bd.values())
            ref_pct = 100 * bd["refusal"] / total_ctx if total_ctx else 0
            row += f"  {ref_pct:>7.1f}%"
        print(row)
    print()

# ── 7. GDN Invariance: verify flatness across quant levels ────────────────────

print("=" * 70)
print("TABLE 6 — GDN Invariance Check: Qwen3.5 4k→16k slope across quant levels")
print("(is the GDN slope invariant to quantization precision?)")
print("=" * 70)
print()

for task in TASKS:
    slopes = {}
    baselines = {}
    for q in QUANTS:
        key = ("Qwen3.5", q)
        if key not in data:
            continue
        vals = [get_accuracy(data[key], task, c) for c in CTXS]
        baselines[q] = vals[0]
        slopes[q] = round(vals[-1] - vals[0], 1) if None not in vals else None
    print(f"  {task}:")
    print(f"    Baselines (4k acc): " + " | ".join(f"{q}: {baselines.get(q, 'N/A'):.1f}%" for q in QUANTS if q in baselines))
    print(f"    Slopes (Δ 4k→16k): " + " | ".join(f"{q}: {slopes.get(q, 'N/A'):+.1f}pp" for q in QUANTS if q in slopes))
    slope_vals = [v for v in slopes.values() if v is not None]
    if len(slope_vals) > 1:
        slope_range = round(max(slope_vals) - min(slope_vals), 1)
        print(f"    Slope variance (max−min): {slope_range:.1f}pp  ← {'INVARIANT ✓' if slope_range < 5 else 'NOT INVARIANT ✗'}")
    print()

# ── 8. Quant sensitivity summary (for paper Table 1) ──────────────────────────

print("=" * 70)
print("TABLE 7 — Quantization Sensitivity: Global Accuracy by Precision")
print("(Average accuracy across all tasks and contexts)")
print("=" * 70)
print()

for model in MODELS:
    print(f"  {model}:")
    print(f"    {'Quant':<6} " + " | ".join(f"{t:>8}" for t in TASKS) + " | {'Overall':>8}")
    print("    " + "-" * 52)
    for q in QUANTS:
        key = (model, q)
        if key not in data:
            continue
        task_avgs = []
        row = f"    {q:<6} "
        for task in TASKS:
            accs = [get_accuracy(data[key], task, c) for c in CTXS]
            accs = [a for a in accs if a is not None]
            avg = round(sum(accs) / len(accs), 1) if accs else None
            task_avgs.append(avg)
            row += f"  {avg:>6.1f}%" if avg else "   N/A  "
        overall = round(sum(task_avgs) / len(task_avgs), 1) if task_avgs else None
        row += f"  {overall:>6.1f}%" if overall else "   N/A  "
        print(row)
    print()

print("=" * 70)
print("Analysis complete.")
print("=" * 70)