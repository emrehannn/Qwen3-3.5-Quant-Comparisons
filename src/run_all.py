"""
Batch runner for all benchmarks with pause/resume support.
Tracks state in JSON file for crash recovery and graceful shutdown.

Key design decisions:
  - Q8 models now run everything despite expose and readme, flash attention + batching saves us from OOM
    long-context KV cache at Q8 weight size — see expose.md).
  - Q4 and Q3 models run all three benchmarks including NeedleBench.
  - Depth sweep is explicit: 5 10 30 50 70 90 percent.
  - NeedleBench runs 4k, 8k, and 16k context. Q8 models included via flash attention.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

from state import BenchmarkState, setup_signal_handlers, STATE_FILE, COMPLETED_DIR

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Model configurations
# Each entry: (display_name, model_path, allowed_benchmarks)
#

# On an RTX 4060 8 GB, a Q8 model (~4.3 GB weights) leaves ~3.7 GB for the
# Running perplexity and GSM8K at ctx=2048 is safe at any quant level.
# ---------------------------------------------------------------------------
MODELS = [
    # Qwen3-4B (pure Transformer)
    ("Qwen3-4B-Q8", "models/Qwen3-4B-Instruct-2507-Q8_0.gguf",
     ["perplexity", "gsm8k", "needlebench"]), 
    ("Qwen3-4B-Q4", "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
     ["perplexity", "gsm8k", "needlebench"]),
    ("Qwen3-4B-Q3", "models/Qwen3-4B-Instruct-2507-UD-Q3_K_XL.gguf",
     ["perplexity", "gsm8k", "needlebench"]),

    # Qwen3.5-4B (GDN hybrid)
    ("Qwen3.5-4B-Q8", "models/Qwen3.5-4B-Q8_0.gguf",
     ["perplexity", "gsm8k", "needlebench"]), 
    ("Qwen3.5-4B-Q4", "models/Qwen3.5-4B-Q4_K_M.gguf",
     ["perplexity", "gsm8k", "needlebench"]),
    ("Qwen3.5-4B-Q3", "models/Qwen3.5-4B-Q3_K_M.gguf",
     ["perplexity", "gsm8k", "needlebench"]),
]

# ---------------------------------------------------------------------------
# Benchmark specs
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "perplexity": {
        "script":   "src/perplexity.py",
        "args":     ["--limit", "1000"],
        "ctx":      2048,
        "est_time": 15,
    },
    "gsm8k": {
        "script":   "src/gsm8k.py",
        "args":     ["--limit", "250"],
        "ctx":      2048,
        "est_time": 15,
    },
    "needlebench": {
        "script": "src/needlebench.py",
        # Context lengths: 4k, 8k, and 16k.
        # --depths: explicit sweep matching the depth-ablation design.
        "args": [
            "--context-lengths", "4096", "8192", "16384",
            "--depths", "5", "10", "30", "50", "70", "90",
            "--num-samples", "15",
            "--n-batch", "256",
            "--needles-per-sample", "5", 
        ],
        "ctx":      16384,   # n_ctx = max(context_lengths) passed to the script
        "est_time": 90,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_duration(minutes: float) -> str:
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = int(minutes // 60)
    mins  = int(minutes % 60)
    return f"{hours}h{mins:02d}m"


def build_queue() -> list[tuple]:
    """
    Return all (config_name, model_path, benchmark, spec) tuples,
    filtered by each model's allowed_benchmarks list.
    Prioritizes Q8 NeedleBench runs first for VRAM debugging.
    """
    priority_queue = []
    standard_queue = []
    
    for config_name, model_path, allowed in MODELS:
        for benchmark, spec in BENCHMARKS.items():
            if benchmark in allowed:
                item = (config_name, model_path, benchmark, spec)
                
                # Push Q8 NeedleBench to the absolute front of the line
                if "Q8" in config_name and benchmark == "needlebench":
                    priority_queue.append(item)
                else:
                    standard_queue.append(item)
                    
    return priority_queue + standard_queue


def verify_model_file(model_path: str) -> tuple[bool, str]:
    path = Path(model_path)
    if not path.exists():
        return False, "File not found"
    size_gb = path.stat().st_size / (1024 ** 3)
    if size_gb < 1.0:
        return False, f"File too small ({size_gb:.2f} GB) — likely incomplete download"
    return True, f"Valid ({size_gb:.2f} GB)"


def print_progress_header(state: BenchmarkState):
    stats     = state.get_stats()
    bar_width = 40
    filled    = int(bar_width * stats["percent"] / 100)
    bar       = "█" * filled + "░" * (bar_width - filled)

    print("\n" + "=" * 70)
    print("BATCH BENCHMARK RUNNER — Pause/Resume Enabled")
    print("=" * 70)
    print(f"State file  : {STATE_FILE.absolute()}")
    print(f"Started     : {state.started_at}")
    print(f"Last updated: {state.last_updated}")
    print(f"\nProgress: [{bar}] {stats['percent']:.1f}%")
    print(f"  Completed : {stats['completed']}/{stats['total']}")
    print(f"  Remaining : {stats['remaining']}")
    if state.in_progress:
        print(f"  Running   : {state.in_progress['config']} / {state.in_progress['benchmark']}")

    if stats["completed"] > 0 and stats["remaining"] > 0:
        start_dt = datetime.fromisoformat(state.started_at)
        elapsed  = (datetime.now() - start_dt).total_seconds() / 60
        avg_time = elapsed / stats["completed"]
        eta_min  = avg_time * stats["remaining"]
        eta_dt   = datetime.now() + timedelta(minutes=eta_min)
        print(f"  ETA       : {format_duration(eta_min)} ({eta_dt.strftime('%H:%M')})")
    print("=" * 70)


def print_benchmark_list(state: BenchmarkState):
    queue = build_queue()
    print("\n" + "=" * 70)
    print("BENCHMARK QUEUE")
    print("=" * 70)
    for idx, (config_name, model_path, benchmark, spec) in enumerate(queue, 1):
        if state.is_completed(config_name, benchmark):
            status = "✓ DONE"
        elif (state.in_progress
              and state.in_progress["config"]    == config_name
              and state.in_progress["benchmark"] == benchmark):
            status = "▶ RUNNING"
        else:
            status = "○ PENDING"
        est = spec.get("est_time", 30)
        print(f"  {idx:2d}. [{status:8s}] {config_name:20s} / {benchmark:12s} (~{est}m)")
    stats = state.get_stats()
    print("-" * 70)
    print(f"  Total: {stats['total']} | Completed: {stats['completed']} | "
          f"Remaining: {stats['remaining']}")
    print("=" * 70)


def run_benchmark(config_name: str, model_path: str, benchmark: str,
                  spec: dict, state: BenchmarkState) -> bool:
    result_file = RESULTS_DIR / f"{config_name}_{benchmark}.json"
    state.set_in_progress(config_name, benchmark)

    start_time = datetime.now()
    est_time   = spec.get("est_time", 30)

    print(f"\n[START] {config_name} / {benchmark}")
    print(f"  Model    : {model_path}")
    print(f"  n_ctx    : {spec.get('ctx', '?')}")
    print(f"  Args     : {' '.join(spec.get('args', []))}")
    print(f"  Est. time: ~{est_time} min")
    print(f"  Started  : {start_time.strftime('%H:%M:%S')}")

    cmd = [
        sys.executable,
        spec["script"],
        model_path,
        "--output", str(result_file),
    ]
    
    # NeedleBench natively computes the max ctx, passing --n-ctx will crash it!
    if benchmark != "needlebench":
        cmd.extend(["--n-ctx", str(spec["ctx"])])
        
    cmd.extend(spec.get("args", []))

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        last_lines = []
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            last_lines.append(line)
            if len(last_lines) > 5:
                last_lines.pop(0)
            if any(x in line.lower() for x in ["%", "it/s", "eta", "progress"]):
                print(f"\r  {line[:80]:<80}", end="", flush=True)
            else:
                print(f"\n  [LOG] {line}")

        process.wait()

        if process.returncode == 0:
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"\n  ✓ Completed in {format_duration(elapsed)}")
            state.mark_completed(config_name, benchmark)
            return True
        else:
            print(f"\n  ✗ Failed (return code {process.returncode})")
            error_msg = "\n".join(last_lines[-3:]) if last_lines else "Unknown error"
            state.log_error(config_name, benchmark, error_msg)
            return False  # DO NOT halt — continue with remaining benchmarks

    except Exception as e:
        print(f"\n  ✗ Exception: {e}")
        state.log_error(config_name, benchmark, str(e))
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch benchmark runner with auto pause/resume")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh, ignoring any saved state")
    args = parser.parse_args()

    queue = build_queue()

    if args.fresh:
        print("Starting fresh (ignoring saved state)...")
        state = BenchmarkState()
        state.remaining = [f"{c}_{b}" for c, _, b, _ in queue]
        for f in RESULTS_DIR.glob("*.json"):
            if f.name != "state.json":
                f.unlink()
        for f in COMPLETED_DIR.glob("*.json"):
            f.unlink()

    elif STATE_FILE.exists():
        state = BenchmarkState.load()
        if state is None:
            print("Corrupt state file — starting fresh...")
            state = BenchmarkState()
            state.remaining = [f"{c}_{b}" for c, _, b, _ in queue]
        else:
            print(f"Resuming from saved state ({len(state.completed)} completed)")
            if state.in_progress:
                print(f"  Previous run interrupted at: "
                      f"{state.in_progress['config']} / {state.in_progress['benchmark']}")
                print("  This benchmark will be restarted.")
                state.in_progress = None

    else:
        print("No previous state found — starting fresh run...")
        state = BenchmarkState()
        # Check for already-completed result files so we can skip them.
        for config_name, _, benchmark, _ in queue:
            key = f"{config_name}_{benchmark}"
            completed_file = COMPLETED_DIR / f"{key}.json"
            result_file    = RESULTS_DIR    / f"{key}.json"
            if completed_file.exists():
                state.completed.append(key)
            elif result_file.exists():
                state.completed.append(key)
                result_file.rename(completed_file)
            else:
                state.remaining.append(key)
        if state.completed:
            print(f"  Found {len(state.completed)} already-completed benchmarks")
        state.save()

    setup_signal_handlers(state)
    print_benchmark_list(state)
    print_progress_header(state)

    for config_name, model_path, benchmark, spec in queue:
        if state.is_completed(config_name, benchmark):
            continue

        valid, msg = verify_model_file(model_path)
        if not valid:
            print(f"\n[SKIP] {config_name} — {msg}: {model_path}")
            state.log_error(config_name, benchmark, f"Invalid model file: {msg}")
            continue

        run_benchmark(config_name, model_path, benchmark, spec, state)
        print_progress_header(state)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    stats = state.get_stats()
    print("\n" + "=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)
    print(f"Total    : {stats['total']}")
    print(f"Completed: {stats['completed']}")
    print(f"Failed   : {len(state.errors)}")
    print(f"Finished : {datetime.now().isoformat()}")

    if state.errors:
        print("\nErrors:")
        for err in state.errors:
            print(f"  - {err['config']}/{err['benchmark']}: {err['error'][:100]}")

    print("\nCompleted results:")
    for f in sorted(COMPLETED_DIR.glob("*_*.json")):
        print(f"  ✓ {f.name} ({f.stat().st_size/1024:.1f} KB)")

    if stats["completed"] == stats["total"] and not state.errors:
        STATE_FILE.unlink(missing_ok=True)
        print(f"\n✓ All benchmarks complete. State file removed.")


if __name__ == "__main__":
    main()