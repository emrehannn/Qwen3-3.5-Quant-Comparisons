    Methodology & Implementation Summary
    
This document outlines the specific technical refinements and experimental constraints implemented in the comparative analysis of Qwen3 (Transformer) and Qwen3.5 (GDN Hybrid) architectures.

1. Quantization Strategy
To evaluate architectural resilience to weight compression, three distinct quantization levels were selected using the GGUF format:

Baseline (Q8_0): 8-bit quantization serves as the high-fidelity proxy for the original model weights, establishing the "Gold Standard" performance and control baseline.

Moderate Compression (Q4_K_M): A standard 4-bit quantization representing the common "sweet spot" for local LLM deployment on consumer hardware.

Extreme Compression (Q3_K_XL / Q3_K_M): ~3-bit quantization intended to stress-test the mathematical stability of the hidden state and attention mechanisms. Note: Q3 variants differ slightly by scheme (UD-Q3_K_XL for Qwen3, Q3_K_M for Qwen3.5) due to upstream availability; both target approximately 3-bit precision.

2. Long-Context Evaluation (OpenCompass NeedleBench)
Initial testing protocols utilized a custom synthetic Needle-In-A-Haystack benchmark. To ensure standardized academic validity and prevent models from exploiting synthetic anomaly-detection shortcuts, the methodology was upgraded to use the official, rigorously vetted OpenCompass NeedleBench dataset (Li et al., 2024).

2.1 Task Typology
The evaluation consists of three distinct tasks testing increasing levels of cognitive load:

S-RT (Single-Needle Retrieval): A pure retrieval task where a single target fact is inserted into the haystack.

M-RT (Multi-Needle Retrieval): Tests sustained attention by scattering multiple needles (5 per sample) throughout the document, requiring the model to successfully locate all of them.

M-RS (Multi-Fact Reasoning): The most complex task. Derivations are inserted as needles, forcing the model to retrieve multiple disparate facts and logically combine them to answer a complex question.

2.2 Scoring
All three tasks use a unified composite_retrieval_score = max(levenshtein_soft_score, predicted_coverage_score, substr_score), with correct = True when score ≥ 0.5. This is format-agnostic — concise extractions (e.g. "Voyager of the Stars") and full template-format answers both score correctly without penalizing either architecture's output style.

Haystack content is held constant across depth conditions for each sample by design (rng reseeded per depth loop with seed 42), ensuring that observed accuracy differences reflect needle position rather than haystack variation.

2.3 The Experimental Grid
To ensure robust statistical validity and high granularity, all models are evaluated across a rigid combinatorial grid:

Context Lengths: 4k, 8k, and 16k tokens.

Depths: Needles are embedded exactly at 5%, 10%, 30%, 50%, 70%, and 90% of the document depth.

Scale: 15 independent trials per depth configuration per context length, resulting in 810 evaluations per model configuration (4,860 total long-context evaluations across 6 model configurations).

3. Hardware & Memory Optimizations (RTX 4060 8GB)
Executing a 16,000-token context window on an 8GB consumer GPU alongside a 4.4GB 8-bit model weight requires extreme memory optimization. The following techniques were implemented to prevent Out-of-Memory (OOM) failures:

Flash Attention: Explicitly enabled to dynamically optimize the KV cache memory footprint, reducing the quadratic memory scaling of the self-attention mechanism at 16k context.

Headless Execution (TTY): The Linux display server (X11/Wayland) was bypassed, reclaiming ~900 MiB of system VRAM exclusively for the LLM runner.

Chunked Prefill: The prompt processing batch size was strictly bounded (n_batch=256). This prevents catastrophic VRAM spikes during the initial tokenization and encoding of the massive 16,000-token prompt.

4. Evaluation Metrics
Positional Ablation: Accuracy is mapped across six specific depths (5%, 10%, 30%, 50%, 70%, 90%) to detect "Recurrent Memory Decay" (the hypothesis that GDN architectures overwrite or degrade early sequence information at long distances). For M-RS visualization, avg_score (continuous composite score) is used instead of binary accuracy to produce smoother depth curves at the lower baseline (~60%).

The Capability Cliff (Delta Analysis): We calculate the Q8 → Q3 Accuracy Drop per task (GSM8K, S-RT, M-RT, M-RS). This isolates the architectural vulnerability to compression, filtering out the model's base intelligence.

Transparency in OOM: If a specific context-depth configuration triggers a catastrophic memory failure, it is explicitly logged and labeled as "OOM" in the analysis rather than being quietly treated as 0% accuracy. This distinguishes hardware limitations from cognitive architectural failures.

5. Generation Constraints
"Thinking mode" was explicitly disabled on both models (no_thinking=True via the system prompt) to ensure outputs reflect base instruction-following capability without extended chain-of-thought generation, enabling a fair architectural comparison.
