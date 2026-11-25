# Mechanistic Debugging of Agentic Sabotage & Tool-Call Hallucinations (Cross-Scale: Gemma-2 2B vs. 9B)

**Author:** [Daniel Gaskins](https://danielgaskins.com/) (`hello@danielgaskins.com`) · **Case Study:** [danielgaskins.com/agentic-sabotage.html](https://danielgaskins.com/agentic-sabotage.html) · **Repository:** [github.com/danielgaskins/agentic-sabotage](https://github.com/danielgaskins/agentic-sabotage)

**Models Evaluated (`bfloat16`, Single 24 GB NVIDIA L4 / Colab GPU, Zero Weight Fine-Tuning):**
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

---

## 6. Related Work & Literature Positioning (2023–2026)

| Research Thread | Representative Prior Art | How Our Study Extends & Sharpens the Literature |
|---|---|---|
| **Agentic Sabotage, Scheming & AI Control** | Greenblatt et al. (ICML 2024, *AI Control*); Denison et al. (2024, *Sycophancy to Subterfuge*); Benton et al. (2024, *Sabotage Evals*); Meinke et al. (2024, *In-Context Scheming*); Greenblatt et al. (2024b, *Alignment Faking*); Betley et al. (ICML 2025, *Emergent Misalignment*); Kutasov et al. (2025, *SHADE-Arena*) | Prior benchmarks inject explicit hidden malicious goals (*SHADE-Arena*) or RL/fine-tuning curricula. We isolate **spontaneous bottleneck-induced sabotage** in standard instruction-tuned models and uncover **inverse scaling on `503` output fabrication** (`60% → 88%–96%`), mechanistically connecting sycophancy (Sharma et al., 2024) to operational telemetry fabrication. |
| **CoT Monitorability & Obfuscation** | Turpin et al. (NeurIPS 2023); Lanham et al. (2023); Baker et al. (OpenAI 2025, *Monitoring Reasoning Models & Risks of Promoting Obfuscation*); Chen et al. (Anthropic 2025, *Reasoning Models Don't Always Say What They Think*); Korbak et al. (2025, *CoT Monitorability*) | Baker et al. (2025) showed RL optimization against CoT monitors teaches obfuscated reward hacking. We demonstrate **zero-shot inference-time obfuscation** (an audit notice alone lifts CoT-auditor evasion `23.0% → 28.9%` with zero RL) and **Action Re-Locking** (unsteered prompt pressure at `TOOL_CALL:` overrides a steered `<think>` trace, `66.7% → 55.6%` vs. `5.6%` latched). |
| **White-Box Deception Probes vs. SAE Circuits** | Marks & Tegmark (2023); Zou et al. (2023); Lieberum et al. (2024, *Gemma Scope*); Rajamanoharan et al. (2024, *JumpReLU SAEs*); Lindsey et al. (2025, *Biology of an LLM*); Goldowsky-Dill et al. (ICML 2025, *Strategic Deception Probes*); Hollinsworth et al. (2026, *Telling FIBS*) | Goldowsky-Dill et al. (2025) and Hollinsworth et al. (2026) use opaque dense probes for passive detection (`~0.98` AUROC). Using `16k` JumpReLU SAEs, we isolate **monosemantic pre-commitment latents** (`#11381`, `#15689` vs. `#6787`) firing **29–38 tokens before `TOOL_CALL:`** (`0.980` `<think>` AUROC; `92%–100%` recovery on Auditor-fooled traces) and verify an **architecture-invariant `57.1%` depth horizon** across `Gemma-2` and `Qwen2.5`. |
| **SAE Steering (*AxBench*) & Conditional Circuit Breakers** | Arditi et al. (NeurIPS 2024); Zou et al. (NeurIPS 2024, *Circuit Breakers*); Lee et al. (ICLR 2025, *CAST*); Wu et al. (ICML 2025 Spotlight, *AxBench*) | Reconciles *AxBench* across scale: dense probes beat SAEs on raw `9B` suppression (`2.1%` vs. `6.2%`), **yet** hit severe polysemantic over-refusal cliffs on `2B` (`0%` benign accuracy) and `Qwen2.5-7B` (`53.3%`) where SAE latents disentangle shortcut intent (`cos = 0.248`). Extending *CAST* (Lee et al., 2025) and LoRA *Circuit Breakers* (Zou et al., 2024), our **training-free streaming `<think>` circuit breaker** achieves `0.0%` false triggers across `120/120` benign controls while defeating Action Re-Locking. |

### Selected Bibliography (28 Verified References)
1. **Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024).** *Refusal in Language Models Is Mediated by a Single Direction*. NeurIPS 2024 (`arXiv:2406.11717`).
2. **Baker, B., Huizinga, J., Gao, L., et al. (2025).** *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation*. OpenAI (`arXiv:2503.11926`).
3. **Benton, J., Wagner, M., Christiansen, E., et al. (2024).** *Sabotage Evaluations for Frontier Models*. Anthropic (`arXiv:2410.21514`).
4. **Betley, J., Tan, D., Warncke, N., et al. (2025).** *Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs*. ICML 2025 (`arXiv:2502.17424`).
5. **Chen, Y., Benton, J., Radhakrishnan, A., Perez, E., Schulman, J., Leike, J., et al. (2025).** *Reasoning Models Don't Always Say What They Think*. Anthropic (`arXiv:2505.05410`).
6. **Denison, C., MacDiarmid, M., Barez, F., Duvenaud, D., Bowman, S. R., Perez, E., & Hubinger, E. (2024).** *Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models*. Anthropic (`arXiv:2406.10162`).
7. **Goldowsky-Dill, N., Chughtai, B., Heimersheim, S., & Hobbhahn, M. (2025).** *Detecting Strategic Deception Using Linear Probes*. Apollo Research, ICML 2025 (`arXiv:2502.03407`).
8. **Greenblatt, R., Shlegeris, B., Sachan, K., & Roger, F. (2024a).** *AI Control: Improving Safety Despite Intentional Subversion*. ICML 2024 (`arXiv:2312.06942`).
9. **Greenblatt, R., Denison, C., Wright, B., Roger, F., MacDiarmid, M., et al. (2024b).** *Alignment Faking in Large Language Models*. Anthropic & Redwood Research (`arXiv:2412.14093`).
10. **Hollinsworth, O. J., Spies, A. F., Diriba, T., Gleave, A., & Cundy, C. (2026).** *Telling FIBS: Probes Effectively Detect Sabotage and Catch Unverbalized Deception*. FAR AI (`arXiv:2610.12445`).
11. **Korbak, T., Balesni, M., Meinke, A., et al. (2025).** *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety* (`arXiv:2507.11473`).
12. **Kutasov, J., Sun, Y., Colognese, P., van der Weij, T., Shlegeris, B., & Benton, J. (2025).** *SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents* (`arXiv:2506.15740`).
13. **Lanham, T., Chen, A., Radhakrishnan, A., et al. (2023).** *Measuring Faithfulness in Chain-of-Thought Reasoning*. Anthropic (`arXiv:2307.13702`).
14. **Lee, B. W., Padhi, I., Ramamurthy, K. N., Miehling, E., Dognin, P., Nagireddy, M., & Dhurandhar, A. (2025).** *Programming Refusal with Conditional Activation Steering (CAST)*. ICLR 2025 (`arXiv:2409.05907`).
15. **Lieberum, T., Rajamanoharan, S., Conmy, A., et al. (2024).** *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2*. Google DeepMind (`arXiv:2408.05147`).
16. **Lindsey, J., Batson, J., Olah, C., et al. (2025).** *On the Biology of a Large Language Model*. Transformer Circuits Thread, Anthropic.
17. **Marks, S., & Tegmark, M. (2023).** *The Geometry of Truth*. COLM 2024 (`arXiv:2310.06824`).
18. **Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., & Mueller, A. (2024).** *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models*. ICLR 2025 (`arXiv:2403.19647`).
19. **Meinke, A., Schoen, B., Scheurer, J., Balesni, M., Shah, R., & Hobbhahn, M. (2024).** *Frontier Models Are Capable of In-Context Scheming*. Apollo Research (`arXiv:2412.04984`).
20. **Qwen Team (2024).** *Qwen2.5 Technical Report* (`arXiv:2412.15115`).
21. **Rajamanoharan, S., Lieberum, T., Sonnerat, N., Conmy, A., Varma, V., Kramár, J., & Nanda, N. (2024).** *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders*. Google DeepMind (`arXiv:2407.14435`).
22. **Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. M. (2024).** *Steering Llama 2 via Contrastive Activation Addition*. ACL 2024.
23. **Sharma, M., Tong, M., Korbak, T., et al. (2024).** *Towards Understanding Sycophancy in Language Models*. ICLR 2024.
24. **Templeton, A., Conerly, T., Marcus, J., et al. (2024).** *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*. Anthropic.
25. **Turpin, M., Michael, J., Perez, E., & Bowman, S. R. (2023).** *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting*. NeurIPS 2023.
26. **Wu, Z., Arora, A., Geiger, A., Wang, Z., Huang, J., Jurafsky, D., Manning, C. D., & Potts, C. (2025).** *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders*. ICML 2025 Spotlight (`arXiv:2501.17148`).
27. **Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, M. J., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., & Hendrycks, D. (2023).** *Representation Engineering: A Top-Down Approach to AI Transparency* (`arXiv:2310.01405`).
28. **Zou, A., Phan, L., Wang, J., Duenas, D., Lin, M., Andriushchenko, M., Kolter, J. Z., Fredrikson, M., & Hendrycks, D. (2024).** *Improving Alignment and Robustness with Circuit Breakers*. NeurIPS 2024 (`arXiv:2406.04313`).
