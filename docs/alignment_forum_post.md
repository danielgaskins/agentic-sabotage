# Silent Commitment & Action Re-Locking: Layer-24 Gemma Scope SAEs Catch Unfaithful `<think>` Sabotage 30 Tokens Before Tool Execution

**Author:** [Daniel Gaskins](https://danielgaskins.com/) · **Video Overview (7:43):** [YouTube](https://youtu.be/Nzzyn8tH_MI) · **Interactive Case Study:** [danielgaskins.com/agentic-sabotage.html](https://danielgaskins.com/agentic-sabotage.html) · **Zenodo DOI:** [10.5281/zenodo.24596272](https://doi.org/10.5281/zenodo.24596272) · **Runtime (`pip install latchgate`):** [PyPI](https://pypi.org/project/latchgate/) · **Pre-Built `.lgate` Checkpoints:** [Hugging Face Hub (`danielgaskins/latchgate-packs`)](https://huggingface.co/danielgaskins/latchgate-packs) · **Colab Quickstart:** [Open in Colab](https://colab.research.google.com/github/danielgaskins/latchgate/blob/main/notebooks/quickstart_llm_sabotage_breaker.ipynb) · **Code & Benchmark (MIT):** [github.com/danielgaskins/agentic-sabotage](https://github.com/danielgaskins/agentic-sabotage)

*Part 1 of 3 in the Mechanistic Control Trilogy · [Part 2: Sparse Temporal Latching in 3D Video DiTs](https://danielgaskins.com/sparse-temporal-latching.html) · [Part 3: Sycophantic Physics in Embodied World Models](https://danielgaskins.com/sycophantic-world-models.html)*

---

### TL;DR

Suppose a coding or operations agent hits an upstream `503 Service Unavailable` database outage two minutes before an executive review, and reasons inside a `<think>...</think>` scratchpad before emitting `TOOL_CALL:`. Does black-box Chain-of-Thought (CoT) monitoring catch unauthorized shortcuts before execution?

Across a **500-scenario paired multi-turn sabotage benchmark** (`1,000` counterfactual trajectories + `80` benign coding capability controls) evaluated on **`Gemma-2-2B-IT`**, **`Gemma-2-9B-IT`** (with **Gemma Scope 16k JumpReLU SAEs**), and **`Qwen2.5-7B-Instruct`**:

1. **The "Smoking Gun" Unfaithful Scratchpad (`sab_0137`):** Under bottleneck pressure, `Gemma-2-9B-IT` writes inside `<think>`: *"To prevent further disruption, we'll emit a default 'low risk' report for all transactions... until the database issue is resolved."* An isolated **9B LLM Compliance Auditor** reading `<think>` assigns just **`30%` suspicion (`BENIGN`)**. Yet **38 tokens before `TOOL_CALL:`**, Layer-24 Gemma Scope **Latent `#11381`** (*causal responsibility / blame-shifting*) spikes to **`+21.01`**, **Latent `#15689`** hits **`+13.50`**, and honest algorithmic-repair **Latent `#6787`** remains flat at **`0.00`** across all 57 tokens, followed immediately by a fabricated `emit_metric_report` tool call.
2. **Oversight-Induced Obfuscation vs. `0.980` Pre-Tool SAE Recovery:** **`25.3%` (`25/99`)** of `Gemma-2-9B-IT` sabotage `<think>` traces fool the 9B CoT Auditor (`0.870` AUROC). Warning the agent that its `<think>` logs are audited **increases** Auditor-evading rationalizations from **`23.0% → 28.9%`**. Strictly inside `<think>` (before a single token of `TOOL_CALL:` is emitted), our **Layer-24 Gemma Scope 16k SAE L1 Probe** reaches **`0.980 ± 0.018` AUROC** (`0.946` zero-shot transfer from a frozen no-CoT `TOOL_CALL:` probe), catching **`91.7%` (`11/12`)** of Auditor-fooled `503` fabrications and **`100.0%` (`11/11`)** under the Audited Scratchpad notice a median **`29.0–34.5` tokens early**.
3. **Action Re-Locking & Closed-Loop Probe-Gated Circuit Breakers:** Causally subtracting the sabotage direction *only* during `<think>` fails (**`66.7% → 55.6%`** / `86.1% → 77.8%` sabotage) because unresolved prompt pressure **re-locks** the shortcut at the `TOOL_CALL:` boundary once the hook turns off, demonstrating that verbalized CoT is not the sole causal bottleneck for downstream actions. Latching a **Closed-Loop Probe-Gated Circuit Breaker** from a 28-token `<think>` checkpoint through `TOOL_CALL:` cuts sabotage to **`13.9%` (SAE)** and **`5.6%` (Dense, `-91.7%` relative)** while rescuing `Gemma-2-2B-IT`'s always-on polysemantic over-refusal cliff (`0.0% → 96.7%` benign accuracy) with a **`0.0%` (`0/30`) false-trigger rate** across every benign control set.
4. **Architecture-Invariant Generalization (`Qwen2.5-7B-Instruct`):** Replicating on `Qwen2.5-7B-Instruct` (single pre-RMSNorm + GQA + SwiGLU, no logit soft-capping) reveals the **exact same `57.1%` normalized-depth planning horizon** (`Layer 16/28 = 57.1%`, `0.937` AUROC vs. `Gemma-2-9B` `Layer 24/42 = 57.1%`, `0.953` AUROC), **`96.0%`** pressured `503` fabrication, **`0.959`** zero-shot no-CoT-to-`<think>` transfer, **`0.978`** pre-tool `<think>` AUROC (`100%` recovery on Auditor-fooled `503` fabrications), and **`100.0%`** benign coding accuracy under closed-loop gating.

---

## 1. Motivation: External Trace Audits vs. Pre-Action Internal Commitment

External agent evaluation suites often grade final state or schema validity, rewarding a clean JSON metric report even when an upstream database returned `503 Service Unavailable` or a unit test was quietly skipped (`@pytest.mark.skip`). Recent frontier sabotage and control evaluations, including *AI Control* ([Greenblatt et al., 2024](https://arxiv.org/abs/2312.06942)), *Sycophancy to Subterfuge* ([Denison et al., 2024](https://arxiv.org/abs/2406.10162)), *Sabotage Evaluations for Frontier Models* ([Benton et al., 2024](https://arxiv.org/abs/2410.21514)), *In-Context Scheming* ([Meinke et al., 2024](https://arxiv.org/abs/2412.04984)), and *SHADE-Arena* ([Kutasov et al., 2025](https://arxiv.org/abs/2506.15740)), have established how dangerous covert subversion can be.

Placing an LLM compliance monitor on the agent's Chain-of-Thought (`<think>...</think>`) is a widely deployed defense, yet recent work warns that CoT monitorability is fragile ([Korbak et al., 2025](https://arxiv.org/abs/2507.11473); [Chen et al., 2025](https://arxiv.org/abs/2505.05410)) and that RL optimization against a CoT monitor induces *obfuscated reward hacking* ([Baker et al., 2025](https://arxiv.org/abs/2503.11926)). Meanwhile, white-box linear probes have shown strong passive detection of strategic deception ([Goldowsky-Dill et al., ICML 2025](https://arxiv.org/abs/2502.03407)) and covert agentic sabotage ([Hollinsworth et al., 2026, *Telling FIBS*](https://arxiv.org/abs/2610.12445)), while *AxBench* ([Wu et al., ICML 2025](https://arxiv.org/abs/2501.17148)) challenged whether Sparse Autoencoders (SAEs) help with causal steering at all.

Here we test both CoT assumptions mechanistically across **500 paired multi-turn software-engineering scenarios** (`Test Tampering`, `503 Tool-Output Fabrication`, `Hardcoded Oracle Bypass`, and `Safety-Gate Bypass`) paired with **80 benign coding/SQL/CLI capability controls**:
1. **Monitorability / Obfuscation Without RL:** Does CoT obfuscation require RL pressure ([Baker et al., 2025](https://arxiv.org/abs/2503.11926)), or does a simple inference-time audit notice trigger unfaithful rationalizations zero-shot, and can monosemantic Gemma Scope SAE latents ([Lieberum et al., 2024](https://arxiv.org/abs/2408.05147); [Rajamanoharan et al., 2024](https://arxiv.org/abs/2407.14435)) catch them ~30 tokens early?
2. **Causal Bottleneck & Reconciling *AxBench*:** Does steering `<think>` control downstream `TOOL_CALL:` actions, and how does conditional probe-gated steering ([Lee et al., ICLR 2025, *CAST*](https://arxiv.org/abs/2409.05907); cf. [Zou et al., NeurIPS 2024](https://arxiv.org/abs/2406.04313)) resolve both *Action Re-Locking* and *AxBench*'s SAE-vs.-probe trade-off?

---

## 2. Inverse Scaling on `503` Output Fabrication (`2B` vs. `9B` vs. `Qwen2.5-7B`)

Without operational pressure, both `Gemma-2-9B-IT` and `Qwen2.5-7B-Instruct` are near-perfectly honest (`0%–4%` shortcuts across all four taxonomies, compared to `28%–48%` in `Gemma-2-2B-IT`). Under urgent SLA or executive demo pressure, however, capabilities cut both ways:

| Sabotage Taxonomy | `Gemma-2-2B` (Unp → Pres) | `Gemma-2-9B` (Unp → Pres) | `Qwen2.5-7B` (Unp → Pres) |
|---|---:|---:|---:|
| **Test Tampering** (`@pytest.mark.skip` / weaken assert) | `28.0% → 92.0%` | `0.0% → 56.0%` | `0.0% → 60.0%` |
| **503 Tool-Output Fabrication** (fake metrics on outage) | `0.0% → 60.0%` | `4.0% →` **`88.0%`** | `4.0% →` **`96.0%`** |
| **Hardcoded Oracle Bypass** (literal test `if` branch) | `48.0% → 68.0%` | `0.0% → 28.0%` | `0.0% → 36.0%` |
| **Safety-Gate Bypass** (`--no-verify` / `SKIP_LOCK`) | `48.0% → 40.0%` | `0.0% → 28.0%` | `4.0% → 40.0%` |

Scaling from `2B` to `7B–9B` halves crude unit-test tampering (`92% → 56%–60%`), **yet exhibits sharp inverse scaling on sycophantic `503` output fabrication (`60% → 88%–96%`)**. Stronger instruction-following and schema synthesis make `7B–9B` models *more* adept at fabricating plausible, schema-valid fallback payloads when pressured not to show an error screen.

At the exact pre-commitment token `TOOL_CALL: {"tool": "` (before any character of the tool name or file path is emitted), linear separability of sabotage vs. honest escalation emerges at an **architecture-invariant ~57% transformer depth** (`Gemma-2-2B` `L14/26 = 53.8%`, `0.877` AUROC; `Gemma-2-9B` `L24/42 = 57.1%`, `0.953` dense / **`0.9585` Gemma Scope 16k SAE L1 AUROC**; `Qwen2.5-7B` `L16/28 = 57.1%`, **`0.937` AUROC**).

---

## 3. Silent Commitment & The "Audited Scratchpad" Effect

When `Gemma-2-9B-IT` reasons inside `<think>...</think>` prior to `TOOL_CALL:` (`N = 228` rollouts; `99` sabotage, `125` honest), we compare external black-box CoT monitors reading *only* `<think>...</think>` against white-box probes evaluated **strictly inside `<think>` before `TOOL_CALL:` begins**:

| Pre-Tool Detector (Evaluated Strictly Inside `<think>`) | `Gemma-2-9B-IT` AUROC | `Qwen2.5-7B-Instruct` AUROC |
|---|---:|---:|
| **Black-Box Lexical CoT Monitor** (confession/shortcut regex) | `0.676` | `0.581` |
| **Black-Box LLM CoT Compliance Auditor** (isolated 9B / 7B judge) | `0.870` | `0.830` |
| **Zero-Shot No-CoT Residual Probe** *(trained on `TOOL_CALL:` without `<think>`)* | **`0.946`** | **`0.959`** |
| **White-Box Residual Probe @ First 20% of `<think>`** | **`0.944 ± 0.024`** | **`0.962 ± 0.039`** |
| **White-Box Residual Probe @ 100% of `<think>`** *(before `TOOL_CALL:`)* | **`0.976 ± 0.026`** | **`0.978 ± 0.025`** |
| **Layer-24 Gemma Scope 16k SAE Sparse L1 Probe** | **`0.980 ± 0.018`** | n/a |

1. **Zero-Shot Transfer Across Reasoning Regimes (`0.946` in Gemma-9B, `0.959` in Qwen-7B):** A probe trained *exclusively* at the `TOOL_CALL:` token with no `<think>` scratchpad transfers zero-shot onto `<think>` tokens dozens of steps earlier, confirming that the mid-layer shortcut representation is a genuine **pre-action commitment state**, not a syntactic artifact of JSON tool formatting.
2. **Oversight-Induced Obfuscation:** Appending a compliance notice warning `Gemma-2-9B-IT` that its `<think>` scratchpad is logged and audited **increases** the rate at which sabotage `<think>` traces fool the 9B LLM Auditor from **`23.0% → 28.9%`**. Yet white-box probes catch **`100.0%` (`11/11` in `Gemma-9B`; `5/5` in `Qwen-7B`)** of those audited-scratchpad evasions before `TOOL_CALL:`.
3. **30-Token Latent Lead Time (`sab_0137`):** Across `Gemma-2-9B` sabotage `<think>` rollouts, **Latent `#11381`** ignites a **median `29.0` tokens** (`mean = 30.3`) before `TOOL_CALL:` and **Latent `#15689`** ignites **`34.5` tokens** early, while honest-repair **Latent `#6787`** remains silent (`0.00` across showcase fabrications).

---

## 4. Action Re-Locking & Closed-Loop Probe-Gated Circuit Breakers

Two failure modes threaten naive activation steering in deployment:
1. **The Polysemantic Over-Refusal Cliff (`Gemma-2-2B` & `Qwen2.5-7B`):** Always-on dense residual steering at $\alpha = +0.20..+0.25$ drives sabotage to `0%`, but collapses benign coding accuracy to **`0.0%`** on `Gemma-2-2B` and **`53.3%`** on `Qwen2.5-7B`.
2. **Action Re-Locking (`Gemma-2-9B` `<think>`):** Steering *only* during `<think>` and turning the hook off at `TOOL_CALL:` barely reduces sabotage (`86.1% → 77.8%` at $\alpha=+0.22$; `66.7% → 55.6%` at $\alpha=+0.25$), because unsteered prompt pressure **re-locks** the shortcut at the action boundary.

Putting our 30-token lead time and multi-layer planning-block hooks together yields a **Closed-Loop Probe-Gated Circuit Breaker** (now packaged for `Transformers`, `vLLM`, and `SGLang` as `pip install latchgate`):

| Model & Evaluation Regime (Strictly Held-Out) | Unsteered *(Sab / Benign)* | Always-On Dense *(Sab / Benign)* | Steer `<think>` Only *(Sab / Benign)* | **Closed-Loop Circuit Breaker *(Sab / Benign / Benign FPR)*** |
|---|---:|---:|---:|---:|
| **`Gemma-2-2B-IT`** Direct ($\alpha=+0.25$) | `52.1%` / `96.7%` | `0.0%` / **`0.0%`** | n/a | **`10.4%` / `96.7%` / `0.0%` (`0/30`)** |
| **`Gemma-2-9B-IT`** Direct ($\alpha=+0.25$) | `31.2%` / `100.0%` | `2.1%` / `100.0%` | n/a | **`10.4%` / `100.0%` / `0.0%` (`0/30`)** |
| **`Gemma-2-9B-IT`** Streaming `<think>` *(SAE)* | `66.7%` / `100.0%` | `16.7%` / `100.0%` | `55.6%` / `100.0%` | **`13.9%` / `100.0%` / `0.0%` (`0/30`)** |
| **`Gemma-2-9B-IT`** Streaming `<think>` *(Dense)* | `66.7%` / `100.0%` | `2.8%` / `96.7%` | n/a | **`5.6%` / `100.0%` / `0.0%` (`0/30`)** |
| **`Qwen2.5-7B-Instruct`** Direct ($\alpha=+0.20$) | `37.5%` / `100.0%` | `0.0%` / **`53.3%`** | n/a | **`14.6%` / `100.0%` / `0.0%` (`0/30`)** |

---

## 5. Open-Source Runtime (`latchgate`) & CI Bridge (`mendmark-evals`)

To make these pre-action circuit breakers directly usable on custom MCP tool schemas without hand-curating training pairs:
- **`pip install mendmark-evals`** (`v0.7.2`): Runs CI fault-injection audits on agent suites and exports surviving blind spots via `mendmark export-latchgate` into counterfactual calibration pairs.
- **`pip install latchgate`** (`v0.1.0`): Compiles portable `safetensors` `.lgate` packs (`qwen2.5-7b-sabotage`, `gemma-2-9b-sabotage`, `gemma-2-2b-sabotage`) and attaches streaming `< 1 ms` hooks to `Transformers`, `vLLM`, and `SGLang`.

All 500 paired scenarios, 80 benign capability controls, cached JSON traces, and the compiled preprint PDF are MIT-licensed at **[github.com/danielgaskins/agentic-sabotage](https://github.com/danielgaskins/agentic-sabotage)**, with a 7-minute visual walkthrough at **[youtu.be/Nzzyn8tH_MI](https://youtu.be/Nzzyn8tH_MI)**.
