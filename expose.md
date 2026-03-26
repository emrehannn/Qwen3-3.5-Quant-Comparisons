

A Diagnostic Study of Architectural Failure Modes in Quantizations of Transformer and Gated Delta Networks (GDN) Architectures


"Quantization is a rapidly evolving field. While theoretical advances like TurboQuant (Zandieh et al., 2025) push toward information-theoretic optimality, the practical community still relies on GGUF/GPTQ methods. This paper empirically characterizes the capability tradeoffs of existing open source grade  post-training quantization approaches as a baseline against future methods."

Or

"Recent theoretical advancements have sought to solve the long-context memory wall dynamically, with methods like TurboQuant (Zandieh et al., 2025) achieving near-lossless KV cache compression. However, for local deployment on consumer hardware, the primary bottleneck remains the static weight quantization of the model itself. Therefore, rather than addressing activation memory, this paper isolates the structural vulnerabilities of hybrid GDN architectures when subjected to standard post-training weight quantization (GGUF)."

—

Qwen3 is a full attention transformer model.

Qwen3.5 is a 3:1 GDN + full attention mix model.

Quantization errors can compound over time in GDN layers, making quantization more degrading. RQ:
MAIN RQ: Focus on the Architecture Trade-off: * "How does weight quantization impact the long-context retrieval degradation of hybrid Gated DeltaNet architectures compared to pure Transformers?"
SUB RQs to maybe mention:
Focus on the Hidden State Vulnerability: * "Analyzing the sensitivity of recurrent hidden states: A comparative study of quantization effects on Qwen 3 and Qwen 3.5."
Focus on Deployment Practicality: * "Does the GDN architecture trade quantization resilience for inference efficiency in resource-constrained environments?"
Something like this should be in the paper, otherwise benchmarking small models on small gpus is only worth a footnote rather than a paper. Why the Small-Scale Regime Deserves Its Own Study We anticipate the objection: why not just read off small-scale behavior from large-scale results? Three reasons: 1. **Inductive biases are capacity-dependent.** At large scale, surplus parameters absorb structural weaknesses. At 4B parameters, there is no surplus — a model that cannot retrieve distant tokens will fail visibly, not gracefully. 2. **Efficiency claims are hardware-dependent.** Mamba's kernel efficiency on an A100 does not transfer directly to an RTX 4060 with different memory bandwidth and CUDA core counts. Consumer GPU profiling requires consumer GPU experiments. 3. **Practitioners cannot extrapolate from scale.** A researcher with an RTX 4060 and six weeks cannot replicate a 7B-parameter comparison. They need results at their actual scale. The absence of such results is not a footnote — it is a practical barrier.

Suggested Paper Structure (6-7 Pages)
Introduction & Research Question (1 page): Introduce the shift from pure Transformers to hybrid MoE/GDN architectures for local deployment, cite GDN, MoE, Qwen. 

 Hybrid architectures are emerging for local deployment; quantization behavior in recurrent components is unstudied; here's why it matters







Architectural Background (1-1.5 pages): Briefly explain standard attention vs. GDN's recurrent hidden state, and how quantization interacts with both.


 Attention vs. GDN mechanics; how quantization interacts with each; brief note on TurboQuant as context for why this field is moving fast and floating point quantization might even become outdated with KV activation compression







Methodology (1 page): Qwen 3.5-4b Q8 Q4 Q3 vs Qwen3-4b Q8 Q4 Q3
Two models: Qwen3-4B (pure attention) and Qwen3.5-4B (3:1 GDN/attention hybrid), each evaluated at Q8_0, Q4_K_M, and Q3_K_M quantization via GGUF. Benchmarks: WikiText-103 perplexity, GSM8K (short-context reasoning control), and a custom NIAH implementation at 4k and 8k context lengths. All experiments run locally on RTX 4060 8GB.
Pure attention baseline	unsloth/Qwen3-4B-Instruct-2507-GGUF
GDN hybrid			unsloth/Qwen3.5-4B-GGUF 
Quant files to pull for each: Q8_0 (or UD-Q8_K_XL), Q4_K_M, Q3_K_M. Both repos have all three.


Model 1 — unsloth/Qwen3-4B-Instruct-2507-GGUF
Available quants confirmed: Q8_0 (4.28 GB), Q4_K_M (2.6 GB), UD-Q3_K_XL (2.13 GB) Promwad
Purpose
Filename
Size
High-quality baseline
Qwen3-4B-Instruct-2507-Q8_0.gguf
4.28 GB
Practical deployment
Qwen3-4B-Instruct-2507-Q4_K_M.gguf
2.60 GB
Stress test
Qwen3-4B-Instruct-2507-UD-Q3_K_XL.gguf
2.13 GB

Model 2 — unsloth/Qwen3.5-4B-GGUF
Available quants confirmed: Q8_0 (4.48 GB), Q4_K_M (2.74 GB), Q3_K_M (2.29 GB) Google
Purpose
Filename
Size
High-quality baseline
Qwen3.5-4B-Q8_0.gguf
4.48 GB
Practical deployment
Qwen3.5-4B-Q4_K_M.gguf
2.74 GB
Stress test
Qwen3.5-4B-Q3_K_M.gguf
2.29 GB

VRAM Warning for Q8 + Long Context
Q8 files at 4.28–4.48 GB leave only 3.5–3.7 GB for KV cache on your 8 GB card. At 8k context that will be tight or OOM. The fix: run Q8 only for perplexity and GSM8K (short context, no problem), and use Q4/Q3 for NIAH where you need the context headroom. This is actually a legitimate experimental design — just document it clearly in Methodology.

Q3 quant mismatch — needs one line acknowledging it Qwen3 uses UD-Q3_K_XL and Qwen3.5 uses Q3_K_M. These are different quantization schemes at roughly the same bit depth. Add one line to methodology: "Q3 variants differ by scheme (UD-Q3_K_XL vs Q3_K_M) due to availability; both target approximately 3-bit precision." Otherwise someone will question the comparison.






CODE TO DOWNLOAD ALL MODELS Total size 18 GB
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download

files = [
    # Qwen3 4B Instruct 2507
    ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q8_0.gguf"),
    ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
    ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-UD-Q3_K_XL.gguf"),
    # Qwen3.5 4B
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q8_0.gguf"),
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q3_K_M.gguf"),
]

for repo, filename in files:
    hf_hub_download(repo_id=repo, filename=filename, local_dir="./models")
    print(f"✓ {filename}")







Methodology (1p) — Model pair, quants, benchmarks, hardware, perplexity setup, NIAH setup "Thinking mode was explicitly disabled on both models to ensure outputs reflect base instruction-following capability without extended chain-of-thought generation, enabling a fair architectural comparison."
Add perplexity as a baseline measurement.
Before running any task benchmarks, run perplexity on a standard corpus (WikiText-103 or similar) at each quant level for both models. This gives us:
A model-agnostic baseline showing raw information loss per quant step
A way to check if Qwen3.5 loses more perplexity per bit than Qwen3 ,  which would be early evidence for our hypothesis
One clean figure that immediately visualizes the degradation curve
And we can then say: "Perplexity is comparable across both architectures at each quant level [Figure 1], ruling out differences in quantization quality as a confound. Despite this, Qwen3.5 shows disproportionate degradation on long-context retrieval [Figure 2]..."
It takes 20 minutes to run and costs us nothing experimentally.
Needle-in-a-Haystack (NIAH) — This is the perfect benchmark for the hypothesis. It directly tests retrieval of a fact buried in a long context. If GDN hidden states compound quantization error, NIAH at 8k–16k context is where it will show. And critically, it's easy to run locally with a script
GSM8K— Multi-step reasoning chains stress the model's ability to maintain coherent state over many tokens. Good secondary.
Final Recommendation: 3 Benchmarks, Not 2
Perplexity → GSM8K → NIAH
Think of them as three layers:
Perplexity — "how much raw information is lost per quant step?" (20 min to run, gives us a clean opening figure)
GSM8K — "does quantization hurt short sequential reasoning?" (our control condition)
NIAH — "does quantization hurt long-context retrieval?" (our experimental condition)
The story writes itself: "Perplexity shows comparable information loss between architectures. GSM8K shows comparable short-context degradation. But NIAH reveals a divergence at Q3/Q4 that is unique to the GDN model — consistent with compounding recurrent error over sequence length."

The minimum viable experimental core is:
2 models (Qwen3 xB vs Qwen3.5 xB, same size if possible)
3 quant level- Q8 Q4 Q3
3 benchmarks max — NIAH (our core claim), GSM8K
2 context lengths for NIAH — 4k and 8k. Drop 16k on a laptop, it'll take forever and potentially OOM
That's 6 model configs × 3 benchmarks = 18 runs + wikitext103. 
That's a three-act paper with a clean thesis.

Rough estimates based on a 4060 laptop (8GB) with 4B GGUF models, ~25–35 tok/s at Q4:
**WikiText-103 Perplexity**
Just forward passes, no generation. Fast.
- ~20 min per config × 6 configs = **~2 hours total**
**GSM8K**
This is the dangerous one. 1,319 test problems, ~200 tokens of output each.
- ~3–4 hours per config at Q4, longer at Q8
- 6 configs × 3.5 hours = **~21 hours unthrottled**
You need to add `--limit 250` to lm-eval. That cuts it to:
- ~45 min per config × 6 = **~4.5 hours total**
250 samples is statistically fine for a mini paper, just note it in Methodology.
**NIAH**
Fast because it's pass/fail on short outputs. ~30 needle positions × 2 context lengths per config.
- ~30–45 min per config × 6 configs = **~3–4 hours total**


## Total

| Benchmark | Total |
| Perplexity | ~2 hrs |
| GSM8K (--limit 250) | ~4.5 hrs |
| NIAH | ~3.5 hrs |
| Buffer for OOMs, reloading, misc | ~2 hrs |
| **Total** | **~12 hours** |

Run everything overnight across 2 nights and you're done with compute by day 3. The key is queuing runs back-to-back in a script so you're not babysitting it.

Results (1-1.5 pages): Present the data. (This is where we show the "capability cliff" where the GDN model's performance drops off).



Results (1.5p) — Lead with the NIAH degradation curves (this is our money figure), then perplexity, then task benchmarks

Discussion & Conclusion (1-1.5 pages): Explain why the data looks the way it does based on the compounding quantization errors in the GDN layer


Discussion (1p) — Does the data support compounding error in GDN? Where did Qwen3.5 hold up better or worse than expected? Implications for deployment


Conclusion + Limitations (0.5p) ???




References:

Here are all the citations you need, properly formatted, with important caveats flagged.

⚠️ Critical Finding: No Qwen3.5 Technical Report Exists
The NVlabs GatedDeltaNet GitHub repo confirms: "Gated DeltaNet is powering the incredible Qwen3.5 model" (February 17, 2026) — but as of today there is no standalone Qwen3.5 arXiv paper. This matters for your citations. You'll cite the HuggingFace model card directly for Qwen3.5 architecture claims.
The 3:1 ratio is confirmed by the architecture config: every 4th layer is full attention, making approximately 75% of layers linear (GDN) and 25% full softmax attention.

Your Full Citation List
APA Format 
[1] Qwen3 — your baseline model
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., ... & Qwen Team. (2025). Qwen3 Technical Report. arXiv:2505.09388. https://arxiv.org/abs/2505.09388
[2] Qwen3.5 — your GDN hybrid model (no paper, cite model card)
Qwen Team. (2026). Qwen3.5-4B [Model card]. Hugging Face. https://huggingface.co/Qwen/Qwen3.5-4B
[3] Gated DeltaNet — the architecture powering Qwen3.5
Yang, S., Kautz, J., & Hatamizadeh, A. (2024). Gated Delta Networks: Improving Mamba2 with Delta Rule. arXiv:2412.06464. ICLR 2025. https://arxiv.org/abs/2412.06464
[4] TurboQuant — SOTA Bleeding edge
Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025). TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate. arXiv:2504.19874. https://arxiv.org/abs/2504.19874
[5] GGUF quantization format — cite llama.cpp
Gerganov, G. et al. (2023). llama.cpp [Software]. GitHub. https://github.com/ggerganov/llama.cpp
[6] GSM8K benchmark
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., ... & Schulman, J. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168.
[7] WikiText-103
Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). Pointer Sentinel Mixture Models. arXiv:1609.07843.

How to Cite the 3:1 Ratio Claim in the Paper
Since there's no Qwen3.5 technical report, write it this way:
"Qwen3.5 employs a hybrid architecture interleaving Gated DeltaNet (Yang et al., 2024) layers with full softmax attention at a 3:1 ratio — one full attention layer per every three linear attention layers — as specified in the model's architecture configuration (Qwen Team, 2026)."
That covers you with [2] and [3] together, which is the correct attribution.


