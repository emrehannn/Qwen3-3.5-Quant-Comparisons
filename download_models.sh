#!/bin/bash
# Alternative model downloader using wget with resume support
# Use this if the Python downloader has issues

set -e

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

echo "=========================================="
echo "Model Download Script (wget fallback)"
echo "=========================================="
echo "Target directory: $MODELS_DIR"
echo ""

# Base URL for HuggingFace
HF_BASE="https://huggingface.co"

# Models to download: (repo_id filename)
declare -a MODELS=(
    "unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q8_0.gguf"
    "unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    "unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-UD-Q3_K_XL.gguf"
    "unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q8_0.gguf"
    "unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf"
    "unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q3_K_M.gguf"
)

total=${#MODELS[@]}
completed=0
failed=0

for entry in "${MODELS[@]}"; do
    repo_id=$(echo "$entry" | cut -d' ' -f1)
    filename=$(echo "$entry" | cut -d' ' -f2)
    
    filepath="$MODELS_DIR/$filename"
    
    # Check if already downloaded and valid (>1GB)
    if [ -f "$filepath" ]; then
        size=$(stat -c%s "$filepath" 2>/dev/null || echo "0")
        size_mb=$((size / 1024 / 1024))
        
        if [ "$size_mb" -gt 1000 ]; then
            echo "✓ $filename already exists (${size_mb} MB)"
            ((completed++))
            continue
        else
            echo "⚠ $filename exists but is too small (${size_mb} MB), re-downloading..."
        fi
    fi
    
    # Construct download URL
    url="$HF_BASE/$repo_id/resolve/main/$filename"
    
    echo ""
    echo "[$((completed + failed + 1))/$total] Downloading $filename"
    echo "  From: $url"
    echo "  To: $filepath"
    echo "  Press Ctrl+C to pause, resume with same command"
    
    # Download with infinite retries and resume support
    attempt=0
    while true; do
        ((attempt++))
        
        if wget --progress=bar:force -c -O "$filepath" "$url" 2>&1; then
            # Verify download
            size=$(stat -c%s "$filepath" 2>/dev/null || echo "0")
            size_mb=$((size / 1024 / 1024))
            
            if [ "$size_mb" -gt 1000 ]; then
                echo "✓ $filename downloaded successfully (${size_mb} MB)"
                ((completed++))
                break
            else
                echo "⚠ Download seems incomplete (${size_mb} MB), retrying..."
                rm -f "$filepath"
            fi
        else
            echo "✗ Attempt $attempt failed, retrying in 10s..."
            sleep 10
        fi
    done
done

echo ""
echo "=========================================="
echo "DOWNLOAD SUMMARY"
echo "=========================================="
echo "Completed: $completed/$total"

# Calculate total size
total_size=$(du -sh "$MODELS_DIR" 2>/dev/null | cut -f1)
echo "Total downloaded: $total_size"

if [ "$completed" -eq "$total" ]; then
    echo ""
    echo "✓ ALL MODELS DOWNLOADED SUCCESSFULLY"
    exit 0
else
    echo ""
    echo "✗ Some downloads failed. Re-run this script to retry."
    exit 1
fi
