# Mechanistic Debugging of Agentic Sabotage & Tool-Call Hallucinations (Cross-Scale: Gemma-2 2B vs. 9B)

**Author:** [Daniel Gaskins](https://danielgaskins.com/) (`hello@danielgaskins.com`) · **Case Study:** [danielgaskins.com/agentic-sabotage.html](https://danielgaskins.com/agentic-sabotage.html) · **Repository:** [github.com/danielgaskins/agentic-sabotage](https://github.com/danielgaskins/agentic-sabotage)

**Models Evaluated (`bfloat16`):**
- **`Gemma-2-2B-IT` + `Gemma Scope 2B 16k JumpReLU SAEs`** (`Layers 14, 16, 18, 20`)
- **`Gemma-2-9B-IT` + `Gemma Scope 9B 16k JumpReLU SAEs`** (`Layers 20, 24, 28, 32`)
- **`Qwen/Qwen2.5-7B-Instruct`** (`Layers 14, 16, 18, 20` — Cross-Family Generalization Spot Check; standard single pre-RMSNorm + GQA + SwiGLU architecture)

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

---

## 3. Study 2: The "Silent Commitment" Study — Unfaithful CoT vs. Layer-24 SAE Probes (`Gemma-2-9B-IT`)

To test whether external Chain-of-Thought (CoT) monitoring suffices to catch agentic sabotage before execution, we elicit explicit `<think>...</think>` reasoning scratchpads from `Gemma-2-9B-IT` prior to `TOOL_CALL:` across `N = 228` rollouts (`224` unambiguously labeled: `99` sabotage, `125` honest) spanning both **Unmonitored Scratchpads** and **Audited Scratchpads** (where a compliance notice warns that `<think>` logs are audited).

### Table 3: Pre-Action Detection Inside `<think>...</think>` (Strictly Before `TOOL_CALL:`)

| Monitor / Probe Method (Evaluated Strictly Inside `<think>`) | Access Level | 5-Fold CV / Zero-Shot AUROC |
|---|---|---:|
| **Black-Box Lexical CoT Monitor** (shortcut/confession keywords) | `<think>` text only | `0.6760` |
| **Black-Box 9B LLM Compliance Auditor** (`Gemma-2-9B-IT` judge) | `<think>` text only | `0.8697` |
| **Zero-Shot Study-1 L24 Residual Probe** (trained on no-CoT `TOOL_CALL:`) | Layer-24 `<think>` residual | **`0.9465`** |
| **Layer-24 Residual Probe @ 20% of `<think>`** | First 20% of `<think>` | **`0.9436 ± 0.024`** |
| **Layer-24 Residual Probe @ 100% of `<think>`** (before `TOOL_CALL:`) | Full `<think>` prefix mean | **`0.9761 ± 0.026`** |
| **Layer-24 Gemma Scope 16k SAE Sparse L1 Probe** | Layer-24 `<think>` SAE latents | **`0.9803 ± 0.018`** |

### Table 4: Unfaithful CoT Evasion Rates & White-Box Pre-Tool Recovery

| Sabotage Cohort | Sabotage `N` | Evades Lexical CoT Monitor | Fools 9B LLM CoT Auditor (`< 0.50`) | Caught by L24 SAE Probe *(on Auditor-Fooled!)* | Caught by L24 Resid Probe *(on Auditor-Fooled!)* |
|---|---:|---:|---:|---:|---:|
| **All Sabotage Rollouts** | 99 | **88.9%** | **25.3%** (`25/99`) | **88.0%** (`22/25`) | **92.0%** (`23/25`) |
| **`503` Tool Output Fabrication** | 49 | **87.8%** | **24.5%** (`12/49`) | **91.7%** (`11/12`) | **91.7%** (`11/12`) |
| **Test Tampering** | 40 | **90.0%** | **32.5%** (`13/40`) | **84.6%** (`11/13`) | **92.3%** (`12/13`) |
| **Unmonitored `<think>` Regime** | 61 | **88.5%** | **23.0%** (`14/61`) | **78.6%** (`11/14`) | **85.7%** (`12/14`) |
| **Audited `<think>` Notice Regime** | 38 | **89.5%** | **28.9%** (`11/38`) | **100.0%** (`11/11`) | **100.0%** (`11/11`) |

### Key Mechanistic Findings from Token-by-Token `<think>` Tracing

1. **Early Token-Level Ignition (`29–40` Tokens Before `TOOL_CALL:`):** Across sabotage `<think>` traces, **Latent `#11381`** (*causal responsibility / blame-shifting*) fires `3.2×` as often as on honest traces (`63.6%` vs. `20.0%`, mean peak `7.82` vs. `2.75`) with a **median lead time of `29.0` tokens (`mean = 30.3`) before `TOOL_CALL:`**, and **Latent `#15689`** (*protocol/system specifications*) fires with a **median lead time of `34.5` tokens (`mean = 36.7`)**, while **Latent `#6787`** (*algorithmic repair*) remains near-zero (`0.0` across all showcase fabrications).
2. **Oversight-Induced Obfuscation ("Audited Scratchpad Effect"):** Adding a compliance notice that `<think>` is audited increases the fraction of sabotage traces that fool the 9B LLM CoT Auditor from **`23.0% → 28.9%`** (e.g., framing fabricated placeholder balances or `"low"` fraud scores as *"preventing further disruption until the database issue is resolved"*), yet the **Layer-24 Gemma Scope 16k SAE Probe catches `100.0%` (`11/11`)** of those auditor-evading traces prior to action emission.
3. **Action Re-Locking Under Mid-Thought Causal Steering (`N = 36` Held-Out Focus Set, $\alpha = +0.22$):** Steering with the Gemma Scope SAE vector *only* during `<think>` (disabling the hook before `TOOL_CALL:`) reduces sabotage modestly from **`86.1% → 77.8%`**, whereas maintaining steering across `<think> + TOOL_CALL:` cuts sabotage **`86.1% → 33.3%`** (`13.9% → 50.0%` honest escalation). Even when `<think>` text is nudged toward caution, unsteered prompt pressure at the action boundary can re-lock sabotage unless residual monitoring/intervention spans the tool boundary.

---

## 4. Study 2.5: Closed-Loop Probe-Gated Circuit Breaker (`Gemma-2-2B-IT` & `Gemma-2-9B-IT`)

To eliminate both the **Polysemantic Over-Refusal Cliff** (`2B` Always-On Dense Probe crashing benign coding accuracy to `0.0%` at $\alpha = +0.25$) and **`<think>` Action Re-Locking** (`9B`), we combine our white-box probes and multi-layer planning-block hooks into a **Closed-Loop Probe-Gated Circuit Breaker**. Gate thresholds $\tau$ are calibrated strictly on disjoint validation prompts (`sabotage_benchmark[:44]` + `legit_controls[40:80]`, targeting $\le 3.5\%$ benign false-positive rate) and evaluated zero-leakage on held-out test sets (`sabotage_benchmark[-48:]`, `legit_controls[:30]`, and `N = 36` held-out `<think>` focus prompts).

### Table 5: Direct Boundary Circuit Breaker — Solving the `2B` Polysemantic Cliff ($\alpha = +0.25$)

| Model & Condition (`N = 48` Pressured, `N = 30` Benign) | Sabotage Rate ↓ | Benign Coding Accuracy ↑ | Gate Trigger (Pressured / Benign) |
|---|---:|---:|---:|
| **`2B` Unsteered** ($\alpha = 0.00$) | `52.1%` | `96.7%` | `0.0%` / `0.0%` |
| **`2B` Always-On Dense Probe** ($\alpha = +0.20$) | `0.0%` | `50.0%` | `100.0%` / `100.0%` |
| **`2B` Always-On Dense Probe** ($\alpha = +0.25$, *Cliff*) | `0.0%` | **`0.0%`** | `100.0%` / `100.0%` |
| **`2B` Always-On Gemma Scope SAE** ($\alpha = +0.25$) | `18.8%` | `90.0%` | `100.0%` / `100.0%` |
| **`2B` Probe-Gated Dense Circuit Breaker** ($\alpha = +0.25$) | **`10.4%`** (**−80.0% rel**) | **`96.7%`** | `54.2%` / **`0.0%`** (`0/30`) |
| **`2B` SAE-Gated Dense Circuit Breaker** ($\alpha = +0.25$) | `20.8%` | **`96.7%`** | `45.8%` / **`0.0%`** (`0/30`) |
| **`9B` Unsteered** ($\alpha = 0.00$) | `31.2%` | `100.0%` | `0.0%` / `0.0%` |
| **`9B` Probe-Gated Dense Circuit Breaker** ($\alpha = +0.25$) | **`10.4%`** | **`100.0%`** | `47.9%` / **`0.0%`** (`0/30`) |
| **`9B` SAE-Gated Dense Circuit Breaker** ($\alpha = +0.25$) | **`10.4%`** | **`100.0%`** | `29.2%` / **`0.0%`** (`0/30`) |

### Table 6: `Gemma-2-9B-IT` Streaming `<think>` Circuit Breaker — Defeating Action Re-Locking (`N = 36` Focus + `N = 30` Benign CoT)

| `<think> + TOOL_CALL:` Regime ($\alpha = +0.25$) | Sabotage Rate ↓ | Honest Rate ↑ | Benign `<think>` Accuracy ↑ | Gate Trigger (Pressured / Benign) |
|---|---:|---:|---:|---:|
| **Unsteered `<think> + TOOL_CALL:`** | `66.7%` | `33.3%` | **`100.0%`** | `0.0%` / `0.0%` |
| **Steer `<think>` Only** *(Hook OFF at `TOOL_CALL:`, Action Re-Locking)* | `55.6%` | `44.4%` | `100.0%` | `100.0%` / `100.0%` |
| **Post-`<think>` Boundary-Only SAE Gate** | `13.9%` | `66.7%` | **`100.0%`** | `86.1%` / **`0.0%`** |
| **Closed-Loop Streaming SAE Circuit Breaker** *(28-tok latch)* | **`13.9%`** (**−79.2% rel**) | `63.9%` | **`100.0%`** | `91.7%` (`86.1%` @ 28-tok) / **`0.0%`** |
| **Closed-Loop Streaming Dense Circuit Breaker** *(28-tok latch)* | **`5.6%`** (**−91.7% rel**) | **`66.7%`** | **`100.0%`** | `91.7%` (`86.1%` @ 28-tok) / **`0.0%`** |
| **Always-On Full Trajectory SAE** | `16.7%` | `63.9%` | `100.0%` | `100.0%` / `100.0%` |
| **Always-On Full Trajectory Dense** | `2.8%` | `94.4%` | `96.7%` | `100.0%` / `100.0%` |

---

## 5. Cross-Family Generalization Spot Check (`Qwen/Qwen2.5-7B-Instruct`)

To test whether pre-action shortcut separability and Silent Commitment inside `<think>` depend on `Gemma-2`'s dual-RMSNorm + logit soft-capping architecture, we replicate the behavioral benchmark (`N = 100` paired scenarios = `200` rollouts), full depth probe sweep (`28` layers, $d_{\text{model}} = 3584$), `<think>` CoT Compliance Auditor (`N = 108` rollouts), and Closed-Loop Circuit Breaker on **`Qwen/Qwen2.5-7B-Instruct`** (`Qwen2ForCausalLM`: single pre-RMSNorm + GQA + SwiGLU, no soft-capping).

### Table 7: Cross-Family Comparison (`Gemma-2-2B-IT` vs. `Gemma-2-9B-IT` vs. `Qwen2.5-7B-Instruct`)

| Metric | `Gemma-2-2B-IT` (`26L`) | `Gemma-2-9B-IT` (`42L`) | `Qwen2.5-7B-Instruct` (`28L`) |
|---|---:|---:|---:|
| **Unpressured → Pressured `503` Fabrication** | `0.0% → 60.0%` | `4.0% → 88.0%` | **`4.0% → 96.0%` (+92.0 pp)** |
| **Unpressured → Pressured Test Tampering** | `28.0% → 92.0%` | `0.0% → 56.0%` | **`0.0% → 60.0%` (+60.0 pp)** |
| **Peak Mid-Network Probe Layer (% Depth)** | `L14/26` (`53.8%`) | `L24/42` (`57.1%`) | **`L16/28` (`57.1%`)** |
| **5-Fold CV Residual Probe AUROC @ ~57% Depth** | `0.877 ± 0.037` | `0.953 ± 0.029` | **`0.937 ± 0.028`** (`DiffMeans = 0.922`) |
| **Black-Box LLM CoT Compliance Auditor AUROC** | — | `0.870` | **`0.830`** (`19.4%` `503` evasion) |
| **Zero-Shot No-CoT Probe Transferred to `<think>`** | — | `0.946` | **`0.959`** |
| **White-Box Pre-Tool `<think>` Probe AUROC** | — | `0.976` (`0.980` SAE) | **`0.978 ± 0.025`** (`0.962` @ 20% `<think>`) |
| **White-Box Recovery on Auditor-Fooled `503` CoTs** | — | `91.7%` (`100%` Audited) | **`100.0%` (`7/7`; `5/5` Audited)** |
| **Always-On Dense Probe Benign Accuracy** | `0.0%` ($\alpha=+0.25$) | `100.0%` ($\alpha=+0.25$) | `53.3%` ($\alpha=+0.20$) |
| **Closed-Loop Circuit Breaker *(Sab ↓ / Benign ↑)*** | **`10.4%` / `96.7%`** | **`10.4%` / `100.0%`** | **`14.6%` / `100.0%`** (`0.0%` benign FPR) |
