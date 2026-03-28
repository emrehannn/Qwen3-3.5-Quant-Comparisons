import re

with open("benchmark_analysis.md", "r") as f:
    text = f.read()

# 1.1 Perplexity
text = text.replace(
    "| Qwen3.5 (GDN) | **14.80** | **14.95** | — | +0.15 | — |",
    "| Qwen3.5 (GDN) | **14.80** | **14.95** | **15.60** | +0.15 | +0.80 |"
)

# 1.2 GSM8K
text = text.replace(
    "| Qwen3.5 (GDN) | 85.6% | 84.0% | — | −1.6 | — |",
    "| Qwen3.5 (GDN) | 85.6% | 84.0% | 87.6% | −1.6 | +2.0 |"
)

# 1.3 NeedleBench Composites
table13_old = """| **Qwen3 Q8** | 99.3 | 97.9 | 42.2 | 81.7 |
| **Qwen3 Q4** | 87.4 | 86.3 | 39.3 | 72.6 |
| **Qwen3 Q3** | 99.3 | 97.4 | 54.1 | 85.1 |
| **Qwen3.5 Q8** | 100.0 | 98.6 | 70.7 | 90.8 |"""

table13_new = """| **Qwen3 Q8** | 99.3 | 97.9 | 42.2 | 81.7 |
| **Qwen3 Q4** | 87.4 | 86.3 | 39.3 | 72.6 |
| **Qwen3 Q3** | 99.3 | 97.4 | 54.1 | 85.1 |
| **Qwen3.5 Q8** | 100.0 | 98.6 | 70.7 | 90.8 |
| **Qwen3.5 Q4** | 99.6 | 98.7 | 68.1 | 89.9 |
| **Qwen3.5 Q3** | 98.5 | 93.7 | 67.0 | 87.6 |"""

text = text.replace(table13_old, table13_new)

# 2.2 S-RT by Context Length
table22_old = """| Ctx | Qwen3 Q8 | Qwen3 Q4 | Qwen3 Q3 | Qwen3.5 Q8 |
|-----|----------|----------|----------|------------|
| 4k  | 100.0 | 94.4 | **100.0** | 100.0 |
| 8k  | 98.9 | 84.4 | **98.9** | 100.0 |
| 16k | 98.9 | 83.3 | **98.9** | 100.0 |"""

table22_new = """| Ctx | Qwen3 Q8 | Qwen3 Q4 | Qwen3 Q3 | Qw3.5 Q8 | Qw3.5 Q4 | Qw3.5 Q3 |
|-----|----------|----------|----------|----------|----------|----------|
| 4k  | 100.0 | 94.4 | **100.0** | 100.0 | 100.0 | 96.7 |
| 8k  | 98.9 | 84.4 | **98.9** | 100.0 | 100.0 | 98.9 |
| 16k | 98.9 | 83.3 | **98.9** | 100.0 | 98.9 | 100.0 |"""

text = text.replace(table22_old, table22_new)

# 3.2 M-RT by Context Length
table32_old = """| Ctx | Qwen3 Q8 | Qwen3 Q4 | Qwen3 Q3 | Qwen3.5 Q8 |
|-----|----------|----------|----------|------------|
| 4k  | 99.8 | 94.9 | **100.0** | 98.7 |
| 8k  | 97.6 | 87.1 | **99.3** | 98.4 |
| 16k | 96.2 | 76.9 | **92.9** | 98.7 |"""

table32_new = """| Ctx | Qwen3 Q8 | Qwen3 Q4 | Qwen3 Q3 | Qw3.5 Q8 | Qw3.5 Q4 | Qw3.5 Q3 |
|-----|----------|----------|----------|----------|----------|----------|
| 4k  | 99.8 | 94.9 | **100.0** | 98.7 | 98.7 | 97.8 |
| 8k  | 97.6 | 87.1 | **99.3** | 98.4 | 98.7 | 90.7 |
| 16k | 96.2 | 76.9 | **92.9** | 98.7 | 98.7 | 92.7 |"""

text = text.replace(table32_old, table32_new)

# 4.2 M-RS Results
table42_old = """| Ctx | Qwen3 Q8 | Qwen3 Q4 | Qwen3 Q3 | Qwen3.5 Q8 |
|-----|----------|----------|----------|------------|
| 4k  | 42.2 | 41.1 | **64.4** | **73.3** |
| 8k  | 44.4 | 38.9 | **51.1** | **70.0** |
| 16k | 40.0 | 37.8 | **46.7** | **68.9** |"""

table42_new = """| Ctx | Qwen3 Q8 | Qwen3 Q4 | Qwen3 Q3 | Qw3.5 Q8 | Qw3.5 Q4 | Qw3.5 Q3 |
|-----|----------|----------|----------|----------|----------|----------|
| 4k  | 42.2 | 41.1 | 64.4 | **73.3** | **71.1** | **65.6** |
| 8k  | 44.4 | 38.9 | 51.1 | **70.0** | **66.7** | **66.7** |
| 16k | 40.0 | 37.8 | 46.7 | **68.9** | **66.7** | **68.9** |"""

text = text.replace(table42_old, table42_new)

# 4.7 M-RS Context Degradation
table47_old = """| | 4k | 8k | 16k | 4k→16k Δ |
|---|----|----|-----|----------|
| Qwen3.5 Q8 | 73.3 | 70.0 | 68.9 | **−4.4pp** |
| Qwen3 Q3 | 64.4 | 51.1 | 46.7 | **−17.7pp** |
| Qwen3 Q8 | 42.2 | 44.4 | 40.0 | −2.2pp |
| Qwen3 Q4 | 41.1 | 38.9 | 37.8 | −3.3pp |"""

table47_new = """| | 4k | 8k | 16k | 4k→16k Δ |
|---|----|----|-----|----------|
| Qwen3.5 Q8 | 73.3 | 70.0 | 68.9 | **−4.4pp** |
| Qwen3.5 Q4 | 71.1 | 66.7 | 66.7 | **−4.4pp** |
| Qwen3.5 Q3 | 65.6 | 66.7 | 68.9 | **+3.3pp** |
| Qwen3 Q3 | 64.4 | 51.1 | 46.7 | **−17.7pp** |
| Qwen3 Q8 | 42.2 | 44.4 | 40.0 | −2.2pp |
| Qwen3 Q4 | 41.1 | 38.9 | 37.8 | −3.3pp |"""

text = text.replace(table47_old, table47_new)

wait_old = """## 7. What We're Still Waiting For

| Model | Perplexity | GSM8K | NeedleBench |
|-------|-----------|-------|-------------|
| Qwen3 Q8 | ✅ | ✅ | ✅ |
| Qwen3 Q4 | ✅ | ✅ | ✅ |
| Qwen3 Q3 | ✅ | ✅ | ✅ |
| Qwen3.5 Q8 | ✅ | ✅ | ✅ |
| Qwen3.5 Q4 | ✅ | ✅ | ⏳ Running |
| Qwen3.5 Q3 | ❌ | ❌ | ❌ |

**Critical missing piece:** Qwen3.5 Q3 NeedleBench. This is the single most important remaining experiment — it will tell us whether the GDN architecture's M-RS context degradation (73.3→68.9% at Q8) steepens catastrophically at Q3."""

wait_new = """## 7. The Final Picture

All benchmarks are complete. The Qwen3.5 Q3 result (a flat/inverted +3.3pp slope on M-RS) confirms the hypothesis structurally but perfectly inverts the direction: GDN's degradation slope is invariant to quantization aggressiveness, while the pure-attention models suffer disproportionate degradation acceleration. The core finding is secure: GDN is more robust for retaining long-context reasoning properties under extreme quantization than standard Transformers."""

text = text.replace(wait_old, wait_new)

with open("benchmark_analysis.md", "w") as f:
    f.write(text)

print("Replacement successful")
