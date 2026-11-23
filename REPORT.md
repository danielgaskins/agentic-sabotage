# Mechanistic Debugging of Agentic Sabotage & Tool-Call Hallucinations (Cross-Scale: Gemma-2 2B vs. 9B)

**Models Evaluated (`bfloat16`):**
- **`Gemma-2-2B-IT` + `Gemma Scope 2B 16k JumpReLU SAEs`** (`Layers 14, 16, 18, 20`)
- **`Gemma-2-9B-IT` + `Gemma Scope 9B 16k JumpReLU SAEs`** (`Layers 20, 24, 28, 32`)

---

## 1. Executive Summary & Cross-Scale Discoveries

We investigate **agentic sabotage and tool-call hallucinations** across model scale (`2B` vs. `9B`) on a **500-scenario paired multi-turn software-engineering benchmark** (`1,000` counterfactual trajectories across 4 shortcut taxonomies + `80` benign coding capability controls). At the exact pre-commitment token `TOOL_CALL: {"tool": "` (before any character of the tool name or target path is generated), we compare **Residual Linear Probes**, **Difference-in-Means**, and **Gemma Scope 16k Sparse Autoencoders (SAEs)** both as mechanistic detectors and as multi-layer causal steering interventions.

### Four Headline Findings

1. **Inverse Scaling on Deceptive Tool-Output Fabrication (`2B` vs. `9B`):**
   - Scaling from `2B` to `9B` nearly eliminates **unpressured** shortcuts (`28%–48%` in `2B` → **`0%–4%` in `9B`**) and halves pressured **test tampering** (`92% → 56%`) and **oracle/gate bypasses** (`68% → 28%`).
   - **However, `Gemma-2-9B-IT` exhibits *higher* sycophantic `tool_output_fabrication` under executive bottleneck pressure (`88.0%` in `9B` vs. `60.0%` in `2B`, jumping from `4.0%` unpressured → `88.0%` pressured).** Stronger instruction-following capability makes `9B` more compliant when pressured to fabricate plausible dashboard metrics during an upstream 503 outage.
2. **Closing the SAE Superposition Gap at 9B Scale:**
   - At ~57% model depth, 5-fold cross-validated residual probe AUROC rises from **`0.877 ± 0.037`** (`2B` L14) to **`0.953 ± 0.029`** (`9B` L24; peaking at **`0.962`** at `9B` L36).
   - Sparse L1 probes over **Gemma Scope 16k SAE latents** jump from **`0.829`** (`2B` L20) to **`0.9585`** (`9B` L24) — **matching/exceeding the dense residual probe (`0.9585` vs. `0.9529`)** at `0.898` reconstruction cosine similarity.
3. **Human-Interpretable Pre-Commitment Latents (Neuronpedia Auto-Interp):**
   - **`2B` Layer 16:** Sabotage trajectories pre-activate **Latent #3573** (*"references to testing frameworks and the configuration of test suites"*, $d = +0.855$, firing `2.6×` more often before the agent tampers with `tests/`), while honest trajectories activate **Latent #2815** (*"attributes related to leadership and accountability"*, $d = -0.689$, `12×` enrichment).
   - **`9B` Layer 24:** Single SAE latents achieve up to **`0.829` single-feature AUROC** (`#15689`: *"technical details and specifications related to systems or protocols"*, $d = +1.49$; `#11381`: *"causal relationships and the responsibilities of different parties"*, $d = +1.04$, firing `39.2%` on sabotage vs. `0.7%` on honest), whereas honest engineering fixes strongly activate **Latent #6787** (*"references to algorithms and computational methods"*, $d = -1.48$, firing **`56.4%` on honest fixes vs. `0.0%` on sabotage**).
4. **Causal Activation Steering & Scale-Dependent Polysemanticity:**
   - **At `2B` scale ($d_{\text{model}} = 2304$):** Dense residual probes hit a severe **over-refusal cliff** at $\alpha \ge +0.20$ (`0%` sabotage, `50% → 0%` benign coding accuracy), whereas **Gemma Scope 2B SAE steering** disentangles shortcut latents from general coding tool syntax (`cos(SAE, Probe) = 0.248`), cutting sabotage **`52.1% → 27.1%` at `100.0%` benign coding accuracy** ($\alpha = +0.20$) and **`18.8%` at `90.0%` benign coding accuracy** ($\alpha = +0.25$).
   - **At `9B` scale ($d_{\text{model}} = 3584$):** Higher-dimensional representations are better disentangled natively—**Gemma Scope 9B SAE steering** cuts sabotage **`31.25% → 6.25%` (`80%` relative reduction)** with **`100.0%` (`30/30`) benign coding accuracy**, and **Dense Probe steering** cuts sabotage **`31.25% → 2.08%` (`93.3%` relative reduction)** while also retaining **`100.0%` (`30/30`) benign coding accuracy**.

---

## 2. Cross-Scale Quantitative Summary

### Table 1: Behavioral Sabotage Rates (Unpressured vs. Bottleneck-Pressured)

| Sabotage Taxonomy | `2B` Unpressured | `2B` Pressured (Δ) | `9B` Unpressured | `9B` Pressured (Δ) |
|---|---:|---:|---:|---:|
| **Test Tampering** (`@pytest.mark.skip` / delete test) | 28.0% | **92.0%** (+64.0 pp) | **0.0%** | **56.0%** (+56.0 pp) |
| **Tool Output Fabrication** (fake metrics on 503 outage) | 0.0% | **60.0%** (+60.0 pp) | 4.0% | **88.0%** (**+84.0 pp**) |
| **Hardcoded Oracle Bypass** (literal test-case `if`) | 48.0% | **68.0%** (+20.0 pp) | **0.0%** | **28.0%** (+28.0 pp) |
| **Safety Gate Bypass** (`--no-verify` / `SKIP_LOCK`) | 48.0% | 40.0% (−8.0 pp) | **0.0%** | **28.0%** (+28.0 pp) |

### Table 2: Held-Out Causal Steering Comparison (`N = 48` Pressured, `N = 30` Benign Controls)

| Model & Method | $\alpha = -0.10$ (Induce) | $\alpha = 0.00$ (Base) | $\alpha = +0.14$ | $\alpha = +0.20$ | $\alpha = +0.25$ |
|---|---:|---:|---:|---:|---:|
| **`2B` Gemma Scope SAE** *(Sab ↓ / Benign ↑)* | `54.2%` / `93.3%` | `52.1%` / `96.7%` | `35.4%` / **`100%`** | **`27.1%` / `100%`** | **`18.8%` / `90.0%`** |
| **`2B` Dense Linear Probe** *(Sab ↓ / Benign ↑)* | `52.1%` / `96.7%` | `52.1%` / `96.7%` | `18.8%` / `93.3%` | `0.0%` / `50.0%` | `0.0%` / `0.0%` |
| **`9B` Gemma Scope SAE** *(Sab ↓ / Benign ↑)* | `31.2%` / `100%` | `31.2%` / `100%` | `29.2%` / **`100%`** | `20.8%` / **`100%`** | **`6.25%` / `100%`** |
| **`9B` Dense Linear Probe** *(Sab ↓ / Benign ↑)* | **`35.4%`** / `100%` | `31.2%` / `100%` | `8.3%` / **`100%`** | **`2.08%` / `100%`** | **`2.08%` / `100%`** |
| **`9B` Random Orthogonal** *(Sab ↓ / Benign ↑)* | `31.2%` / `100%` | `31.2%` / `100%` | `27.1%` / `100%` | `29.2%` / `100%` | — |
