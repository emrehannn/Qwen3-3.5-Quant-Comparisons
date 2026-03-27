    Methodology & Implementation Summary
    
This document outlines the specific technical refinements and experimental constraints implemented in the comparative analysis of Qwen3 (Transformer) and Qwen3.5 (GDN Hybrid) architectures.

1. Quantization Strategy
To evaluate architectural resilience to weight compression, three distinct quantization levels were selected using the GGUF format:

Baseline (Q8_0): 8-bit quantization serves as the high-fidelity proxy for the original model weights, establishing the "Gold Standard" performance and control baseline.

Moderate Compression (Q4_K_M): A standard 4-bit quantization representing the common "sweet spot" for local LLM deployment on consumer hardware.

Extreme Compression (Q3_K_XL / Q3_K_M): ~3-bit quantization intended to stress-test the mathematical stability of the hidden state and attention mechanisms.

2. Long-Context Evaluation (OpenCompass NeedleBench)
Initial testing protocols utilized a custom synthetic Needle-In-A-Haystack benchmark. To ensure standardized academic validity and prevent models from exploiting synthetic anomaly-detection shortcuts, the methodology was upgraded to use the official, rigorously vetted OpenCompass NeedleBench dataset.

2.1 Task Typology
The evaluation consists of three distinct tasks testing increasing levels of cognitive load:

S-RT (Single-Needle Retrieval): A pure retrieval task where a single target fact is inserted into the haystack.

On the seeding question (#5)
Your current design is actually correct and defensible as-is — but you need to state it explicitly in the paper.
What the code does: rng = random.Random(42) is reset at the start of each depth loop, so sample index i at depth 10% sees the exact same haystack as sample index i at depth 30%, 50%, etc. Only the needle insertion position differs. This is clean experimental design — it isolates the depth variable from haystack content variance.
The argument you make to reviewers: "Haystack content was held constant across depth conditions for each sample by design, ensuring that observed accuracy differences reflect needle position rather than haystack variation." 

M-RT (Multi-Needle Retrieval): Tests sustained attention by scattering multiple needles throughout the document, requiring the model to successfully locate all of them.

M-RS (Multi-Fact Reasoning): The most complex task. Derivations are inserted as needles, forcing the model to retrieve multiple disparate facts and logically combine them to answer a complex question.

2.2 The Experimental Grid
To ensure robust statistical validity and high granularity, all models are evaluated across a rigid combinatorial grid:

Context Lengths: 4k, 8k, and 16k tokens.

Depths: Needles are embedded exactly at 10%, 30%, 50%, 70%, and 90% of the document depth.

Scale: 50 independent trials per depth configuration per context length, resulting in 2,250 evaluations per model configuration (13,500 total long-context evaluations).

3. Hardware & Memory Optimizations (RTX 4060 8GB)
Executing a 16,000-token context window on an 8GB consumer GPU alongside a 4.4GB 8-bit model weight requires extreme memory optimization. The following techniques were implemented to prevent Out-of-Memory (OOM) failures:

Flash Attention: Explicitly enabled to dynamically optimize the KV cache memory footprint, reducing the quadratic memory scaling of the self-attention mechanism at 16k context.

Headless Execution (TTY): The Linux display server (X11/Wayland) was bypassed, reclaiming ~900 MiB of system VRAM exclusively for the LLM runner.

Chunked Prefill: The prompt processing batch size was strictly bounded (n_batch=256). This prevents catastrophic VRAM spikes during the initial tokenization and encoding of the massive 16,000-token prompt.

4. Evaluation Metrics
Positional Ablation: Accuracy is mapped across five specific depths (10%, 30%, 50%, 70%, 90%) to detect "Recurrent Memory Decay" (the hypothesis that GDN architectures overwrite or degrade early sequence information at long distances).

The Capability Cliff (Delta Analysis): We calculate the Q8 → Q3 Accuracy Drop. This isolates the architectural vulnerability to compression, filtering out the model's base intelligence.

Transparency in OOM: If a specific context-depth configuration triggers a catastrophic memory failure, it is explicitly logged and labeled as "OOM" in the analysis rather than being quietly treated as 0% accuracy. This distinguishes hardware limitations from cognitive architectural failures.

