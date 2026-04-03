#!/usr/bin/env python3
"""
Fact-check script for NeedleBench paper draft claims.
Run this to verify all numerical claims against the actual database.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path("results/completed")

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

def load_data():
    data = {}
    for key, fname in FILES.items():
        path = BASE / fname
        if path.exists():
            with open(path) as f:
                data[key] = json.load(f)
    return data

def get_accuracy(d, task, ctx):
    for entry in d["tasks"][task]:
        if entry["context_length"] == ctx:
            return entry["accuracy"] * 100
    return None

def get_composite_score(d, task):
    """Get composite score for a task (stored as [0,1] in JSON)."""
    return d.get("composite_scores", {}).get(task, 0) * 100

def calculate_noise_floor(data):
    """
    Calculate the noise floor (standard error) from the data.
    The paper claims σ ≈ 2.8 pp.
    """
    # Collect all individual trial scores across all conditions
    all_scores = []
    
    for key, d in data.items():
        for task in ["S-RT", "M-RT", "M-RS"]:
            for entry in d["tasks"].get(task, []):
                for trial in entry.get("trials", []):
                    # Score is in [0,1], convert to percentage points
                    score = trial.get("score", 0) * 100
                    all_scores.append(score)
    
    if not all_scores:
        return None
    
    # Standard deviation of individual trial scores
    std = np.std(all_scores, ddof=1)
    # Standard error of the mean for n=15 (samples per cell)
    n = 15  # trials per depth per context
    sem = std / np.sqrt(n)
    
    return {
        "std_individual": round(std, 2),
        "sem_mean_n15": round(sem, 2),
        "total_trials": len(all_scores),
        "n_per_cell": n
    }

def verify_finding1_bounded_degradation(data):
    """
    Finding 1: Qwen3.5 (GDN) shows bounded degradation.
    Paper claims: ~-4.4pp drop from 4k to 16k on M-RS at Q8.
    """
    print("\n" + "="*70)
    print("FINDING 1: Bounded Degradation in GDN (Qwen3.5)")
    print("="*70)
    
    q8 = data[("Qwen3.5", "Q8")]
    mrs_4k = get_accuracy(q8, "M-RS", 4096)
    mrs_16k = get_accuracy(q8, "M-RS", 16384)
    delta = mrs_16k - mrs_4k
    
    print(f"M-RS Qwen3.5 Q8: 4k={mrs_4k:.1f}%, 16k={mrs_16k:.1f}%, Δ={delta:+.1f}pp")
    print(f"Paper claims: -4.4pp drop")
    print(f"VERDICT: {'✓ MATCH' if abs(delta - (-4.4)) < 1 else '✗ MISMATCH'}")
    
    return {"delta": delta, "claimed": -4.4}

def verify_finding2_qwen3_accelerated_degradation(data):
    """
    Finding 2: Qwen3 shows accelerated degradation under quantization.
    Paper claims: -3.6pp (Q8) vs -18.0pp (Q4) on M-RT.
    """
    print("\n" + "="*70)
    print("FINDING 2: Precision-Dependent Slope Acceleration (Qwen3)")
    print("="*70)
    
    # M-RT at Q8
    q3_q8 = data[("Qwen3", "Q8")]
    mrt_q8_4k = get_accuracy(q3_q8, "M-RT", 4096)
    mrt_q8_16k = get_accuracy(q3_q8, "M-RT", 16384)
    delta_q8 = mrt_q8_16k - mrt_q8_4k
    
    # M-RT at Q4
    q3_q4 = data[("Qwen3", "Q4")]
    mrt_q4_4k = get_accuracy(q3_q4, "M-RT", 4096)
    mrt_q4_16k = get_accuracy(q3_q4, "M-RT", 16384)
    delta_q4 = mrt_q4_16k - mrt_q4_4k
    
    print(f"M-RT Qwen3 Q8: 4k={mrt_q8_4k:.1f}%, 16k={mrt_q8_16k:.1f}%, Δ={delta_q8:+.1f}pp")
    print(f"Paper claims: -3.6pp drop")
    print(f"VERDICT Q8: {'✓ MATCH' if abs(delta_q8 - (-3.6)) < 1 else '✗ MISMATCH'}")
    
    print(f"\nM-RT Qwen3 Q4: 4k={mrt_q4_4k:.1f}%, 16k={mrt_q4_16k:.1f}%, Δ={delta_q4:+.1f}pp")
    print(f"Paper claims: -18.0pp drop")
    print(f"VERDICT Q4: {'✓ MATCH' if abs(delta_q4 - (-18.0)) < 1 else '✗ MISMATCH'}")

def verify_finding3_task_specificity(data):
    """
    Finding 3: Task-specificity controls (GSM8K sanity check).
    """
    print("\n" + "="*70)
    print("FINDING 3: Task-Specificity Controls")
    print("="*70)
    
    # Check GSM8K files
    for model in ["Qwen3", "Qwen3.5"]:
        for q in ["Q8", "Q4", "Q3"]:
            fname = f"{model}-4B-{q}_gsm8k.json"
            path = BASE / fname
            if path.exists():
                with open(path) as f:
                    gsm = json.load(f)
                acc = gsm.get("accuracy", 0) * 100
                print(f"{model} {q}: GSM8K = {acc:.1f}%")

def verify_finding4_behavioral_divergence(data):
    """
    Finding 4: RQ3 Behavioral divergence (refusal rates).
    Paper claims Qwen3.5: ~14% refusal (Q8, Q4, Q3), 23% (UD-Q3)
    Paper claims Qwen3: fluctuates, grows to 50% at 16k Q4
    """
    print("\n" + "="*70)
    print("FINDING 4: Behavioral Divergence (Refusal Rates)")
    print("="*70)
    
    NOT_FOUND_MARKER = "NOT FOUND"
    
    def refusal_rate(d, task="M-RS"):
        total = 0
        refusals = 0
        for entry in d["tasks"].get(task, []):
            for trial in entry.get("trials", []):
                total += 1
                pred = trial.get("predicted", "")
                if NOT_FOUND_MARKER in str(pred):
                    refusals += 1
        return (refusals / total * 100) if total > 0 else 0
    
    print("\nQwen3.5 Refusal Rates (M-RS):")
    for q in ["Q8", "Q4", "Q3", "UD-Q3"]:
        key = ("Qwen3.5", q)
        if key in data:
            rate = refusal_rate(data[key])
            print(f"  {q}: {rate:.1f}%")
    
    print("\nPaper claims: Qwen3.5 stable ~14% (Q8/Q4/Q3), 23% (UD-Q3)")
    
    print("\nQwen3 Refusal Rates by Context (Q4):")
    for ctx in [4096, 8192, 16384]:
        for entry in data[("Qwen3", "Q4")]["tasks"].get("M-RS", []):
            if entry["context_length"] == ctx:
                total = len(entry.get("trials", []))
                refusals = sum(1 for t in entry.get("trials", []) 
                              if NOT_FOUND_MARKER in str(t.get("predicted", "")))
                rate = refusals / total * 100 if total > 0 else 0
                print(f"  {ctx//1024}k: {rate:.1f}%")
    
    print("\nPaper claims: Qwen3 Q4 grows 42.2% (4k) → 50.0% (16k)")
def verify_finding5_evaluation_artifact(data):
    """
    Finding 5: Evaluation artifact - UD-Q3 overperformance on M-RS.
    Paper claims: 60.7% UD-Q3 vs 48.9% Q8 on M-RS at 4k context.
    (Previously I checked Overall composite - wrong metric!)
    """
    print("\n" + "="*70)
    print("FINDING 5: Evaluation Artifact (UD-Q3 Overperformance)")
    print("="*70)
    
    # Check M-RS specifically at 4k (where the 48.9% claim appears)
    q3_ud = data[("Qwen3", "UD-Q3")]
    q3_q8 = data[("Qwen3", "Q8")]
    
    # Get M-RS accuracy at 4k context
    mrs_ud_4k = get_accuracy(q3_ud, "M-RS", 4096)
    mrs_q8_4k = get_accuracy(q3_q8, "M-RS", 4096)
    
    print(f"Qwen3 M-RS Accuracy at 4k context:")
    print(f"  UD-Q3: {mrs_ud_4k:.1f}%")
    print(f"  Q8:    {mrs_q8_4k:.1f}%")
    print(f"  Diff:  {mrs_ud_4k - mrs_q8_4k:+.1f}pp")
    print(f"\nPaper claims: 60.7% vs 48.9% (+11.8pp)")
    print(f"VERDICT: {'✓ MATCH' if abs((mrs_ud_4k-mrs_q8_4k) - 11.8) < 2 else '✗ MISMATCH'}")
    
    # Also check full M-RS across all contexts (overall M-RS, not Overall composite)
    def avg_mrs_accuracy(d):
        accs = [get_accuracy(d, "M-RS", c) for c in [4096, 8192, 16384]]
        accs = [a for a in accs if a is not None]
        return sum(accs) / len(accs) if accs else 0
    
    mrs_ud_avg = avg_mrs_accuracy(q3_ud)
    mrs_q8_avg = avg_mrs_accuracy(q3_q8)
    
    print(f"\nQwen3 M-RS Average across all contexts:")
    print(f"  UD-Q3: {mrs_ud_avg:.1f}%")
    print(f"  Q8:    {mrs_q8_avg:.1f}%")

def main():
    print("="*70)
    print("NEEDLEBENCH DRAFT FACT-CHECK")
    print("="*70)
    
    data = load_data()
    print(f"Loaded {len(data)} result files")
    
    # Check noise floor claim
    noise = calculate_noise_floor(data)
    if noise:
        print(f"\nNOISE FLOOR ANALYSIS:")
        print(f"  Std of individual scores: {noise['std_individual']:.2f}pp")
        print(f"  SEM (n={noise['n_per_cell']}): {noise['sem_mean_n15']:.2f}pp")
        print(f"  Total trials: {noise['total_trials']}")
        print(f"\nPaper claims: σ ≈ 2.8 pp")
        print(f"VERDICT: {'✓ PLAUSIBLE' if noise['sem_mean_n15'] < 3.5 else '✗ TOO HIGH'}")
    
    # Verify each finding
    verify_finding1_bounded_degradation(data)
    verify_finding2_qwen3_accelerated_degradation(data)
    verify_finding3_task_specificity(data)
    verify_finding4_behavioral_divergence(data)
    verify_finding5_evaluation_artifact(data)
    
    print("\n" + "="*70)
    print("FACT-CHECK COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()

    