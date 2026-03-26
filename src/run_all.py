"""
Batch runner for all benchmarks with pause/resume support.
Tracks state in JSON file for crash recovery and graceful shutdown.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

from state import BenchmarkState, setup_signal_handlers, STATE_FILE, COMPLETED_DIR, RESULTS_DIR

# Model configurations
MODELS = [
    ("Qwen3-4B-Q8", "models/Qwen3-4B-Instruct-2507-Q8_0.gguf"),
    ("Qwen3-4B-Q4", "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
    ("Qwen3-4B-Q3", "models/Qwen3-4B-Instruct-2507-UD-Q3_K_XL.gguf"),
    ("Qwen3.5-4B-Q8", "models/Qwen3.5-4B-Q8_0.gguf"),
    ("Qwen3.5-4B-Q4", "models/Qwen3.5-4B-Q4_K_M.gguf"),
    ("Qwen3.5-4B-Q3", "models/Qwen3.5-4B-Q3_K_M.gguf"),
]

# Benchmark configs
BENCHMARKS = {
    "perplexity": {
        "script": "src/perplexity.py",
        "args": ["--limit", "1000"],
        "ctx": 2048,
        "est_time": 20,  # minutes
    },
    "gsm8k": {
        "script": "src/gsm8k.py",
        "args": ["--limit", "250"],
        "ctx": 2048,
        "est_time": 45,  # minutes
    },
    "niah": {
        "script": "src/niah.py",
        "args": ["--context-lengths", "4096", "8192", "--num-positions", "30"],
        "ctx": 16384,
        "est_time": 35,  # minutes
    },
}

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def format_duration(minutes: float) -> str:
    """Format minutes into human readable string."""
    if minutes < 60:
        return f"{minutes:.0f}m"
    else:
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours}h{mins:02d}m"


def print_progress_header(state: BenchmarkState):
    """Print current progress overview."""
    stats = state.get_stats()
    
    print("\n" + "=" * 70)
    print(f"BATCH BENCHMARK RUNNER - Pause/Resume Enabled")
    print("=" * 70)
    print(f"State file: {STATE_FILE.absolute()}")
    print(f"Started: {state.started_at}")
    print(f"Last updated: {state.last_updated}")
    print("-" * 70)
    
    # Progress bar
    bar_width = 40
    filled = int(bar_width * stats['percent'] / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    print(f"Progress: [{bar}] {stats['percent']:.1f}%")
    print(f"  Completed: {stats['completed']}/{stats['total']} benchmarks")
    print(f"  Remaining: {stats['remaining']}")
    if state.in_progress:
        print(f"  In progress: {state.in_progress['config']} / {state.in_progress['benchmark']}")
    print("=" * 70)
    
    # ETA calculation
    if stats['completed'] > 0 and stats['remaining'] > 0:
        start_time = datetime.fromisoformat(state.started_at)
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        avg_time = elapsed / stats['completed']
        eta_minutes = avg_time * stats['remaining']
        eta = datetime.now() + timedelta(minutes=eta_minutes)
        print(f"Est. remaining: {format_duration(eta_minutes)} (ETA: {eta.strftime('%H:%M')})")
        print("=" * 70)


def print_benchmark_list(state: BenchmarkState):
    """Print full list of benchmarks with completion status."""
    print("\n" + "=" * 70)
    print("BENCHMARK QUEUE")
    print("=" * 70)
    
    queue = build_queue()
    
    for idx, (config_name, model_path, benchmark, spec) in enumerate(queue, 1):
        key = f"{config_name}_{benchmark}"
        
        # Determine status
        if state.is_completed(config_name, benchmark):
            status = "✓ DONE"
            status_color = ""
        elif state.in_progress and state.in_progress["config"] == config_name and state.in_progress["benchmark"] == benchmark:
            status = "▶ RUNNING"
            status_color = ""
        else:
            status = "○ PENDING"
            status_color = ""
        
        model_short = config_name.replace("Qwen3-4B-", "Q3:").replace("Qwen3.5-4B-", "Q3.5:")
        est_time = spec.get("est_time", 30)
        
        print(f"  {idx:2d}. [{status:8s}] {model_short:8s} / {benchmark:12s} (~{est_time}m)")
    
    # Summary stats
    stats = state.get_stats()
    print("-" * 70)
    print(f"  Total: {stats['total']} | Completed: {stats['completed']} | Remaining: {stats['remaining']}")
    print("=" * 70)


def print_current_progress(state: BenchmarkState, step_info: str = ""):
    """Print current step-by-step progress."""
    if state.in_progress:
        config = state.in_progress["config"]
        benchmark = state.in_progress["benchmark"]
        started = datetime.fromisoformat(state.in_progress["started_at"])
        elapsed = datetime.now() - started
        elapsed_mins = elapsed.total_seconds() / 60
        
        print(f"\n[PROGRESS] {config} / {benchmark}")
        print(f"  Elapsed: {format_duration(elapsed_mins)}")
        if step_info:
            print(f"  Step: {step_info}")
        print("-" * 40)


def run_benchmark(config_name: str, model_path: str, benchmark: str, spec: dict, state: BenchmarkState) -> bool:
    """Run a single benchmark with progress tracking."""
    result_file = RESULTS_DIR / f"{config_name}_{benchmark}.json"
    
    # Mark as in progress
    state.set_in_progress(config_name, benchmark)
    
    start_time = datetime.now()
    est_time = spec.get("est_time", 30)
    
    print(f"\n[START] {config_name} / {benchmark}")
    print(f"  Model: {model_path}")
    print(f"  Estimated time: ~{est_time} minutes")
    print(f"  Started at: {start_time.strftime('%H:%M:%S')}")
    
    cmd = [
        sys.executable,
        spec["script"],
        model_path,
        "--output", str(result_file),
        "--n-ctx", str(spec["ctx"]),
    ] + spec["args"]
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        
        # Print output in real-time
        last_lines = []
        for line in process.stdout:
            line = line.rstrip()
            if line:
                last_lines.append(line)
                if len(last_lines) > 5:
                    last_lines.pop(0)
                # Print progress indicators
                if any(x in line.lower() for x in ["%", "it/s", "eta", "progress"]):
                    print(f"\r  {line[:80]:<80}", end="", flush=True)
        
        process.wait()
        
        if process.returncode == 0:
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"\n  ✓ Completed in {format_duration(elapsed)}")
            state.mark_completed(config_name, benchmark)
            return True
        else:
            print(f"\n  ✗ Failed with return code {process.returncode}")
            error_msg = "\n".join(last_lines[-3:]) if last_lines else "Unknown error"
            state.log_error(config_name, benchmark, error_msg)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n  ✗ Timeout after 2 hours")
        state.log_error(config_name, benchmark, "Timeout")
        return False
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        state.log_error(config_name, benchmark, str(e))
        return False


def verify_model_file(model_path: str) -> tuple[bool, str]:
    """Verify model file exists and has valid size (>1GB)."""
    path = Path(model_path)
    
    if not path.exists():
        return False, "File not found"
    
    size_bytes = path.stat().st_size
    size_gb = size_bytes / (1024**3)
    
    # All models should be > 1GB (Q3 is ~2GB, Q4 is ~2.7GB, Q8 is ~4.3GB)
    if size_gb < 1.0:
        return False, f"File too small ({size_gb:.2f} GB) - likely incomplete download"
    
    return True, f"Valid ({size_gb:.2f} GB)"


def build_queue() -> list[tuple[str, str, str, dict]]:
    """Build the full queue of benchmarks to run."""
    queue = []
    for config_name, model_path in MODELS:
        for benchmark, spec in BENCHMARKS.items():
            queue.append((config_name, model_path, benchmark, spec))
    return queue


def main():
    parser = argparse.ArgumentParser(description="Batch benchmark runner with auto pause/resume")
    parser.add_argument("--fresh", action="store_true", help="Start fresh, ignore any saved state")
    args = parser.parse_args()
    
    # Auto-resume by default (unless --fresh is specified)
    if args.fresh:
        print("Starting fresh (ignoring any saved state)...")
        state = BenchmarkState()
        queue = build_queue()
        state.remaining = [f"{c}_{b}" for c, _, b, _ in queue]
        # Clear any existing results
        for f in RESULTS_DIR.glob("*.json"):
            if f.name != "state.json":
                f.unlink()
        for f in COMPLETED_DIR.glob("*.json"):
            f.unlink()
    elif STATE_FILE.exists():
        state = BenchmarkState.load()
        if state is None:
            print("No valid state found, starting fresh...")
            state = BenchmarkState()
            queue = build_queue()
            state.remaining = [f"{c}_{b}" for c, _, b, _ in queue]
        else:
            print(f"Auto-resuming from saved state ({len(state.completed)} completed)")
            if state.in_progress:
                print(f"Previous run interrupted at: {state.in_progress['config']} / {state.in_progress['benchmark']}")
                print("  This benchmark will be restarted (benchmark-level resume)")
                state.in_progress = None
    else:
        print("No previous state found. Starting fresh run...")
        state = BenchmarkState()
        queue = build_queue()
        
        # Check for existing completed files in both folders
        for config_name, model_path, benchmark, _ in queue:
            key = f"{config_name}_{benchmark}"
            # Check if already completed (in completed folder or results folder)
            if (COMPLETED_DIR / f"{key}.json").exists():
                state.completed.append(key)
            elif (RESULTS_DIR / f"{key}.json").exists():
                state.completed.append(key)
                # Move to completed folder
                (RESULTS_DIR / f"{key}.json").rename(COMPLETED_DIR / f"{key}.json")
            else:
                state.remaining.append(key)
        
        if state.completed:
            print(f"  Found {len(state.completed)} already completed benchmarks")
        state.save()
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers(state)
    
    # Print full benchmark list with status
    print_benchmark_list(state)
    
    # Print overall progress header
    print_progress_header(state)
    
    # Build work queue
    queue = build_queue()
    
    # Run benchmarks
    for idx, (config_name, model_path, benchmark, spec) in enumerate(queue, 1):
        key = f"{config_name}_{benchmark}"
        
        # Skip if already completed
        if state.is_completed(config_name, benchmark):
            continue
        
        # Check if model exists and is valid
        valid, msg = verify_model_file(model_path)
        if not valid:
            print(f"\n[SKIP] {config_name} - {msg}: {model_path}")
            state.log_error(config_name, benchmark, f"Invalid model file: {msg}")
            continue
        
        # Run the benchmark
        run_benchmark(config_name, model_path, benchmark, spec, state)
        
        # Print updated progress
        print_progress_header(state)
    
    # Final summary
    stats = state.get_stats()
    print("\n" + "=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)
    print(f"Total benchmarks: {stats['total']}")
    print(f"Completed: {stats['completed']}")
    print(f"Failed: {len(state.errors)}")
    print(f"Finished: {datetime.now().isoformat()}")
    
    if state.errors:
        print("\nErrors:")
        for err in state.errors:
            print(f"  - {err['config']}/{err['benchmark']}: {err['error'][:100]}")
    
    print("\nCompleted results in: results/completed/")
    print("In-progress results in: results/")
    for f in sorted(COMPLETED_DIR.glob("*_*.json")):
        size = f.stat().st_size / 1024
        print(f"  ✓ {f.name} ({size:.1f} KB)")
    
    # Clear state file on successful completion
    if stats['completed'] == stats['total'] and len(state.errors) == 0:
        STATE_FILE.unlink(missing_ok=True)
        print(f"\n✓ All benchmarks complete. State file removed.")


if __name__ == "__main__":
    main()
