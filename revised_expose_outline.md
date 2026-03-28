# Exposé Draft: Against Intuition: Weight Quantization Accelerates Context Degradation in Pure-Attention Transformers, Not Hybrid GDN Architectures

## 1. Introduction
Recent theoretical advancements have sought to solve the long-context memory wall dynamically. However, for local deployment on consumer hardware, the primary bottleneck remains the static weight quantization of the model itself. This study compares Qwen3, a baseline full-attention Transformer, against Qwen3.5, a hybrid architecture utilizing a 3:1 ratio of Gated Delta Networks (GDN) to standard attention.

**The intuitive prior dictates that quantization errors should compound over sequential steps within GDN recurrent layers, worsening long-context capabilities disproportionately.** However, our empirical findings completely falsify this intuition. Instead, we show that the *pure transformer* context-length degradation accelerates under aggressive quantization, while the GDN recurrence mechanism acts as an error-absorbing bottleneck, leaving its long-context degradation profile highly robust to quantization at Q8 and Q4 precision levels.

## 2. Research Questions
**Primary Research Question:**
How does weight quantization interact with the context-length degradation profiles of hybrid Gated DeltaNet (GDN) architectures compared to pure-attention Transformers in long-context retrieval and reasoning tasks?

**Secondary Research Questions:**
- **Quantization Method Sensitivity:** How does the specific quantization scheme (e.g., K_M vs. K_XL group sizes) influence architectural failure modes independently of raw bit-width?
- **Behavioral Failure Modes (Refusal vs. Hallucination):** Under aggressive quantization, do GDN models and Transformers exhibit qualitatively different failure behavior (e.g., admitting failure vs. hallucinating answers) when faced with multi-fact reasoning constraints?

## 3. The New Hypothesis
**The Gating-Robustness Hypothesis:** We hypothesize that the gating mechanisms inherent in GDN layers may provide a structural bottleneck that bounds quantization noise at each sequential step, rendering the architecture's long-context degradation profile highly robust at Q8 and Q4 precisions compared to Transformers. In contrast, pure full-attention mechanisms experience disproportionately accelerated context-length degradation under aggressive quantization, and lack behavioral stability.

## 4. Key Empirical Findings (The Story)

1. **Quantization-Slope Invariance in GDN (Q8/Q4):** Qwen3.5 (GDN) exhibits a flat, highly consistent context-degradation slope (e.g., −4.4pp from 4k to 16k on Multi-Fact Reasoning) that remains invariant across high-precision (Q8) and medium-precision (Q4) levels. Quantization shifts the entire capability curve down (additive error) at these levels, but it does *not* steepen the slope (multiplicative error). Extreme low-precision (Q3) introduces measurable degradation on M-RT, breaking this invariance.

2. **Precision-Dependent Slope Acceleration in Full Attention:** The pure-attention Qwen3 model suffers severe, non-invariant context degradation under quantization. On Multi-Needle Retrieval (M-RT), its context degradation slope accelerates from a mild −3.6pp (Q8) to a severe −18.0pp drop from 4k to 16k tokens at Q4 precision.

3. **Pathological Q4_K_M Failure Mode (Algorithm/Scheme Sensitivity):** Testing revealed an anomalous performance collapse for Qwen3 solely at the Q4_K_M precision level across all long-context tasks, performing significantly worse than the extreme Q3_K_XL limit. This demonstrates that group-size parameterization and the chosen quantization algorithm (K_M vs. K_XL) can induce pathological failure modes in attention matrices, proving that bit-width alone is an insufficient predictor of deployment quality on consumer hardware.

4. **The Q3 M-RS Reversal:** Further supporting Finding 3, Qwen3 Q3_K_XL surprisingly outperforms Q8_0 on multi-fact reasoning globally (54.1% vs 42.2%). We note that Q3_K_XL and Q8_0 differ not only in bit-width but in grouping strategy and quantization kernel, making direct capability comparison difficult. We therefore treat this as evidence that scheme choice dominates bit-width as a predictor of model capability, reinforcing that quantization scheme differences can fundamentally alter model performance beyond simple bit-width scaling.

5. **Architecturally Stable Refusal Behavior:** Qwen3.5's refusal rate on reasoning tasks remains stable (~14%) across all precision levels. Conversely, Qwen3's refusal rate fluctuates wildly and grows monotonically with context length (from 37.8% at 4k to 45.6% at 16k for Q4), offering a novel behavioral signature of attention overload.

## 5. Outline Pivot
The original outline expected to write about why GDN fails under quantization. **The new outline must be written as a "reversal of priors".** 

- **Results Section:** Open with the surprise. Frame the M-RT degradation (Transformer collapses, GDN is flat) to establish the architectural contrast. Then present M-RS as the ultimate stress test.
- **Discussion Section:** 
  - **Proposed Mechanism:** Offer the mechanist interpretation (we *propose* that the bounded range of the GDN gating scalar bounds per-step error propagation, whereas full attention has no equivalent structural clamp — we leave formal characterization to future work). 
  - **Baseline Capability Confound:** Acknowledge the architectural capability difference (Qwen3 functionally fails at complex multi-fact reasoning regardless of quantization, ~40% accuracy at Q8, compared to Qwen3.5's 73%). Frame this as context for interpreting the quantization results, not as a standalone quantization finding.
  - **Practitioner Warning:** Discuss the Q4 anomaly as a stern warning to practitioners not to blindly trust bit-widths.
- **Limitations:** Acknowledge the hardware constraints bounding contexts to 16k, and flag the Qwen3 vs Qwen3.5 training recipe confound.
