"""
State management for pause/resume benchmark system.
Tracks completed benchmarks and in-progress state.
"""
import json
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

STATE_FILE = Path("results/state.json")
RESULTS_DIR = Path("results")
COMPLETED_DIR = RESULTS_DIR / "completed"

# Ensure directories exist
COMPLETED_DIR.mkdir(parents=True, exist_ok=True)


class BenchmarkState:
    """Manages benchmark execution state for pause/resume functionality."""
    
    def __init__(self):
        self.started_at: str = datetime.now().isoformat()
        self.last_updated: str = self.started_at
        self.completed: list[str] = []
        self.in_progress: Optional[dict] = None
        self.remaining: list[str] = []
        self.errors: list[dict] = []
    
    @classmethod
    def load(cls) -> Optional["BenchmarkState"]:
        """Load state from file if it exists."""
        if not STATE_FILE.exists():
            return None
        
        with open(STATE_FILE) as f:
            data = json.load(f)
        
        state = cls()
        state.started_at = data.get("started_at", state.started_at)
        state.last_updated = data.get("last_updated", state.last_updated)
        state.completed = data.get("completed", [])
        state.in_progress = data.get("in_progress")
        state.remaining = data.get("remaining", [])
        state.errors = data.get("errors", [])
        return state
    
    def save(self):
        """Save current state to file."""
        self.last_updated = datetime.now().isoformat()
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(STATE_FILE, "w") as f:
            json.dump({
                "started_at": self.started_at,
                "last_updated": self.last_updated,
                "completed": self.completed,
                "in_progress": self.in_progress,
                "remaining": self.remaining,
                "errors": self.errors,
            }, f, indent=2)
    
    def is_completed(self, config_name: str, benchmark: str) -> bool:
        """Check if a benchmark is already completed."""
        key = f"{config_name}_{benchmark}"
        return key in self.completed
    
    def set_in_progress(self, config_name: str, benchmark: str):
        """Mark a benchmark as currently running."""
        self.in_progress = {
            "config": config_name,
            "benchmark": benchmark,
            "started_at": datetime.now().isoformat(),
        }
        self.save()
    
    def mark_completed(self, config_name: str, benchmark: str):
        """Mark a benchmark as completed and move result to completed folder."""
        key = f"{config_name}_{benchmark}"
        if key not in self.completed:
            self.completed.append(key)
        
        # Remove from remaining
        if key in self.remaining:
            self.remaining.remove(key)
        
        self.in_progress = None
        
        # Move result file to completed folder
        result_file = RESULTS_DIR / f"{key}.json"
        completed_file = COMPLETED_DIR / f"{key}.json"
        
        if result_file.exists():
            result_file.rename(completed_file)
        
        self.save()
    
    def log_error(self, config_name: str, benchmark: str, error: str):
        """Log an error for a benchmark."""
        self.errors.append({
            "config": config_name,
            "benchmark": benchmark,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        total = len(self.completed) + len(self.remaining)
        if self.in_progress:
            total += 1
        
        return {
            "completed": len(self.completed),
            "remaining": len(self.remaining),
            "in_progress": 1 if self.in_progress else 0,
            "total": total,
            "percent": (len(self.completed) / total * 100) if total > 0 else 0,
        }


def setup_signal_handlers(state: BenchmarkState):
    """Setup signal handlers for graceful shutdown."""
    def handle_signal(signum, frame):
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n\n[{sig_name}] Shutdown requested. Saving state...")
        state.save()
        print(f"[{sig_name}] State saved to {STATE_FILE}")
        print(f"[{sig_name}] Run 'python src/run_all.py' to continue")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)