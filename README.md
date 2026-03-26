# Quantization Effects on Transformer vs Gated DeltaNet Architectures

This repository contains the experimental code and benchmarks for comparing quantization degradation between pure Transformer models (Qwen3) and hybrid Gated DeltaNet architectures (Qwen3.5).

## Research Question

**Main RQ:** How does weight quantization impact the long-context retrieval degradation of hybrid Gated DeltaNet architectures compared to pure Transformers?

## Repository Structure

```
├── models/              # Downloaded GGUF models (not in git)
├── results/             # Benchmark outputs (not in git)
│   └── completed/       # Finished benchmark results
├── src/                 # Source code
│   ├── download_models.py   # Model downloader with retry
│   ├── perplexity.py        # WikiText-103 perplexity benchmark
│   ├── gsm8k.py             # GSM8K math reasoning benchmark
│   ├── niah.py              # Needle-in-a-Haystack long-context test
│   ├── run_all.py           # Batch runner with pause/resume
│   ├── state.py             # State management for crash recovery
│   └── analyze.py           # Results analysis and plotting
├── requirements.txt     # Python dependencies
└── expose.md            # Research paper plan and citations
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install llama-cpp-python with CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Install other dependencies
pip install -r requirements.txt
```

### 2. Download Models

```bash
python src/download_models.py
```

This downloads 6 GGUF models (~18GB total):
- Qwen3-4B: Q8_0, Q4_K_M, UD-Q3_K_XL
- Qwen3.5-4B: Q8_0, Q4_K_M, Q3_K_M

### 3. Run Benchmarks

```bash
# Start or resume benchmarks (auto-resumes by default)
python src/run_all.py

# Start fresh (ignore any saved state)
python src/run_all.py --fresh
```

**Pause/Resume:** Press `Ctrl+C` to pause. State is saved automatically. Run the same command later to resume.

### 4. Analyze Results

```bash
python src/analyze.py
```

Generates comparison plots in `results/figures/`.

## Benchmarks

| Benchmark | Purpose | Est. Time |
|-----------|---------|-----------|
| **WikiText-103 Perplexity** | Raw information loss per quant level | ~2 hrs total |
| **GSM8K (250 samples)** | Short-context reasoning control | ~4.5 hrs total |
| **NIAH (4k & 8k context)** | Long-context retrieval (main claim) | ~3.5 hrs total |

**Total runtime:** ~10-12 hours on RTX 4060 8GB

## Hardware Requirements

- **GPU:** NVIDIA GPU with CUDA support (tested on RTX 4060 8GB)
- **RAM:** 16GB+ recommended
- **Storage:** ~20GB free space (18GB models + results)
- **OS:** Linux (tested on Debian/Ubuntu)

## Results Structure

Results are stored in two folders:
- `results/*.json` - In-progress benchmarks
- `results/completed/*.json` - Completed benchmarks (moved automatically)

## Citation

See `expose.md` for full paper structure, methodology details, and citations.

## License

Research code for academic use.
