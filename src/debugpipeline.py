"""
Pre-flight Pipeline Debugger
Runs 1 single sample of every benchmark for every model.
Prints the raw generated text, expected ground truth, and calculated scores.
Uses subprocess to guarantee VRAM is flushed between model loads.
"""
import json
import subprocess
import sys
from pathlib import Path
from needlebench import extract_structured_answer

MODELS = [
    ("Qwen3-4B-Q8", "models/Qwen3-4B-Instruct-2507-Q8_0.gguf"),
    ("Qwen3-4B-Q4", "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
    ("Qwen3-4B-Q3", "models/Qwen3-4B-Instruct-2507-UD-Q3_K_XL.gguf"),
    ("Qwen3.5-4B-Q8", "models/Qwen3.5-4B-Q8_0.gguf"),
    ("Qwen3.5-4B-Q4", "models/Qwen3.5-4B-Q4_K_M.gguf"),
    ("Qwen3.5-4B-Q3", "models/Qwen3.5-4B-Q3_K_M.gguf"),
]

def run_cmd(cmd):
    """Run command silently, but print stderr if it crashes."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed: {' '.join(cmd)}")
        print(result.stderr[-1000:])  # Print last 1000 chars of error
        return False
    return True

def main():
    print("=" * 70)
    print("PIPELINE DEBUGGER: 1 Sample Per Task Per Model")
    print("=" * 70)

    assert extract_structured_answer("</think>reasoning</think><answer>Paris</answer>") == "Paris"
    assert extract_structured_answer("The answer is Paris") == "The answer is Paris"
    assert extract_structured_answer("<answer>NOT FOUND</answer>") == "NOT FOUND"
    print("✓ extract_structured_answer sanity checks passed\n")

    temp_dir = Path("results/debug")
    temp_dir.mkdir(parents=True, exist_ok=True)

    for name, path in MODELS:
        print(f"\n\n{'#' * 70}")
        print(f"MODEL: {name}")
        print(f"{'#' * 70}")

        if not Path(path).exists():
            print(f"[SKIP] Model file missing: {path}")
            continue

        # ------------------------------------------------------------------
        # 1. PERPLEXITY
        # ------------------------------------------------------------------
        ppl_out = temp_dir / f"{name}_ppl.json"
        print("\n[1/3] Running Perplexity (1 chunk)...")
        if run_cmd([sys.executable, "src/perplexity.py", path, "--limit", "5", "--output", str(ppl_out)]):
            with open(ppl_out) as f:
                data = json.load(f)
            print(f"  ✓ Score: {data.get('perplexity', 'N/A')}")

        # ------------------------------------------------------------------
        # 2. GSM8K
        # ------------------------------------------------------------------
        gsm_out = temp_dir / f"{name}_gsm.json"
        print("\n[2/3] Running GSM8K (1 sample)...")
        if run_cmd([sys.executable, "src/gsm8k.py", path, "--limit", "1", "--output", str(gsm_out)]):
            with open(gsm_out) as f:
                data = json.load(f)
            sample = data["results"][0]
            print(f"  Question   : {sample['question'][:100]}...")
            print(f"  Expected   : {sample['ground_truth']}")
            # Using repr() to expose hidden newlines or tabs
            print(f"  Raw Output : {repr(sample['generated_text'])}") 
            print(f"  Extracted  : {sample['predicted']}")
            print(f"  Correct?   : {sample['correct']}")

        # ------------------------------------------------------------------
        # 3. NEEDLEBENCH
        # ------------------------------------------------------------------
        nb_out = temp_dir / f"{name}_nb.json"
        print("\n[3/3] Running NeedleBench (4k ctx, 50% depth, 1 sample)...")
        cmd = [
            sys.executable, "src/needlebench.py", path,
            "--context-lengths", "4096",
            "--depths", "50",
            "--num-samples", "1",
            "--output", str(nb_out)
        ]
        if run_cmd(cmd):
            with open(nb_out) as f:
                data = json.load(f)

            # S-RT
            try:
                srt = data["tasks"]["S-RT"][0]["trials"][0]
                print("\n  --- S-RT (Single Retrieval) ---")
                print(f"  Question : {srt['question']}")
                print(f"  Expected : {srt['expected']}")
                print(f"  Generated: {repr(srt['predicted'])}")
                print(f"  Score    : {srt['score']} (Match: {srt['correct']})")
            except Exception:
                print("  ! Failed to parse S-RT")

            # M-RT
            try:
                mrt = data["tasks"]["M-RT"][0]["trials"][0]
                print("\n  --- M-RT (Multi Retrieval) ---")
                for i, nr in enumerate(mrt["needle_results"]):
                    print(f"  Needle {i+1} Expected : {nr['expected']}")
                    print(f"  Needle {i+1} Generated: {repr(nr['predicted'])}")
                    print(f"  Needle {i+1} F1       : {nr['f1']:.2f}")
            except Exception:
                print("  ! Failed to parse M-RT")

            # M-RS
            try:
                mrs = data["tasks"]["M-RS"][0]["trials"][0]
                print("\n  --- M-RS (Multi Reasoning) ---")
                print(f"  Question : {mrs['question'][:100]}...")
                print(f"  Expected : {mrs['expected']}")
                print(f"  Generated: {repr(mrs['predicted'])}")
                print(f"  Score    : {mrs['score']} (Match: {mrs['correct']})")
            except Exception:
                print("  ! Failed to parse M-RS")

if __name__ == "__main__":
    main()