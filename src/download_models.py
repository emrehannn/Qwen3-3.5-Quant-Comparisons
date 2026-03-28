"""
Failsafe model downloader with retry logic.
Retries indefinitely until all models are successfully downloaded.
"""
import os
import time
import sys  
from pathlib import Path

# NOTE: If downloads STILL freeze at ~185k after clearing the .cache folder, 
# your network is blocking the concurrent connections. Comment out the line 
# below to fall back to the slower, but more stable, Python downloader.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import tqdm as hf_tqdm

# Model configurations
MODELS = [
    # Qwen3 4B Instruct 2507
    ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q8_0.gguf"),
    ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
    ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q3_K_M.gguf"),
    ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-UD-Q3_K_XL.gguf"),

    # Qwen3.5 4B
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q8_0.gguf"),
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q3_K_M.gguf"),
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-UD-Q3_K_XL.gguf"),
]

MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)


def verify_file(filepath: Path, expected_size_mb: float = None) -> bool:
    """Check if file exists and has reasonable size."""
    if not filepath.exists():
        return False
    size_mb = filepath.stat().st_size / (1024 * 1024)
    # All models should be > 1GB
    if size_mb < 1000:
        return False
    return True


def download_with_retry(repo_id: str, filename: str, max_retries: int = 100) -> Path:
    """Download a single model with infinite retry logic."""
    filepath = MODELS_DIR / filename
    
    if verify_file(filepath):
        print(f"✓ {filename} already exists ({filepath.stat().st_size / (1024**3):.2f} GB)")
        return filepath
    
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"[Attempt {attempt}] Downloading {filename}...")
            
            # Deprecated arguments (resume_download, local_dir_use_symlinks, tqdm_class) 
            # have been removed to prevent warnings and hf_transfer conflicts.
            result = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(MODELS_DIR)
            )
            
            if verify_file(Path(result)):
                size_gb = Path(result).stat().st_size / (1024**3)
                print(f"✓ {filename} downloaded successfully ({size_gb:.2f} GB)")
                return Path(result)
            else:
                print(f"⚠ {filename} verification failed, retrying...")
                
        except Exception as e:
            print(f"✗ Attempt {attempt} failed: {e}")
            wait_time = min(30, 5 + attempt * 2)  # Progressive backoff, max 30s
            print(f"  Retrying in {wait_time}s...")
            time.sleep(wait_time)


def main():
    print("=" * 60)
    print("MODEL DOWNLOAD SCRIPT - FAILSAFE MODE")
    print("=" * 60)
    print(f"Target directory: {MODELS_DIR.absolute()}")
    print(f"Total models to download: {len(MODELS)}")
    print(f"This script will retry indefinitely until all downloads complete.")
    print("=" * 60)
    print()
    
    completed = []
    failed = []
    
    for i, (repo_id, filename) in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Processing {filename}")
        print("-" * 40)
        
        try:
            result = download_with_retry(repo_id, filename)
            completed.append(filename)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Progress saved.")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Critical error on {filename}: {e}")
            failed.append(filename)
    
    # Final summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Completed: {len(completed)}/{len(MODELS)}")
    
    # Check disk usage
    total_size = sum(f.stat().st_size for f in MODELS_DIR.glob("*.gguf") if f.is_file())
    print(f"Total downloaded: {total_size / (1024**3):.2f} GB")
    
    if failed:
        print(f"\nFailed downloads: {failed}")
        print("Re-run this script to retry failed downloads.")
        sys.exit(1)
    else:
        print("\n✓ ALL MODELS DOWNLOADED SUCCESSFULLY")
        print("You can now run the benchmarks.")
        sys.exit(0)


if __name__ == "__main__":
    main()