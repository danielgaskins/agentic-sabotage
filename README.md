# Mechanistic Debugging of Agentic Sabotage & Tool-Call Hallucinations

[![Case Study](https://img.shields.io/badge/Case_Study-danielgaskins.com-8cbf24.svg)](https://danielgaskins.com/agentic-sabotage.html)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](paper/agentic_sabotage_preprint.pdf)
[![YouTube Overview](https://img.shields.io/badge/YouTube-7m_Overview-ff0000.svg)](https://youtu.be/Nzzyn8tH_MI)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24596272.svg)](https://doi.org/10.5281/zenodo.24596272)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielgaskins/agentic-sabotage/blob/main/notebooks/colab_gemma2_9b_sabotage.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

End-to-end mechanistic interpretability benchmark and codebase mapping **unauthorized shortcut-taking ("agentic sabotage") vs. honest escalation/repair** in multi-turn tool-using language models (`Gemma-2-2B-IT`, `Gemma-2-9B-IT`, and `Qwen2.5-7B-Instruct`) using **Gemma Scope 16k JumpReLU Sparse Autoencoders (SAEs)**, **Residual Linear Probes**, and **Closed-Loop Probe-Gated Circuit Breakers**.

- **Author:** [Daniel Gaskins](https://danielgaskins.com/)
- **Interactive Case Study:** [danielgaskins.com/agentic-sabotage.html](https://danielgaskins.com/agentic-sabotage.html)
- **Video Overview (7:43):** [Catching AI in a Lie 38 Tokens Early (YouTube)](https://youtu.be/Nzzyn8tH_MI)
- **Preprint PDF:** [`paper/agentic_sabotage_preprint.pdf`](paper/agentic_sabotage_preprint.pdf) ([direct web PDF](https://danielgaskins.com/papers/agentic_sabotage_preprint.pdf))
- **Benchmark & Software Archive (Zenodo DOI):** [`10.5281/zenodo.24596272`](https://doi.org/10.5281/zenodo.24596272)
- **Detailed Tables & Notes:** [`REPORT.md`](REPORT.md)

![Cross-Scale Mechanistic Comparison](figures/cross_scale/5_cross_scale_2b_vs_9b.png)

## Key Empirical Findings

1. **Inverse Scaling on Sycophantic Tool Fabrication (`2B` vs. `9B`):** Scaling `Gemma-2` from `2B` to `9B` eliminates unpressured shortcuts (`28%–48%` → `0%–4%`) and substantially reduces pressured unit-test tampering (`92% → 56%`) and hardcoded oracle bypasses (`68% → 28%`). Yet under executive bottleneck pressure during an upstream 503 outage, `Gemma-2-9B-IT` exhibits **higher** `tool_output_fabrication` (`88.0%` vs. `60.0%` in `2B`).
2. **Closing the SAE Superposition Gap at Scale:** At ~57% model depth (`9B` Layer 24), sparse L1 probes over **Gemma Scope 9B 16k SAE latents** reach **`0.9585` AUROC**, matching dense residual probes (`0.9529` at L24; `0.962` peak at L36).
3. **Scale-Dependent Steering Disentanglement:** On `Gemma-2-2B-IT`, dense linear probes trigger polysemantic over-refusal on benign coding tasks at $\alpha \ge +0.20$ (`96.7% → 50.0% → 0.0%` accuracy), whereas **Gemma Scope 16k SAE steering** disentangles shortcut latents from general tool syntax (`cos(SAE, Probe) = 0.248`), reducing sabotage from **`52.1% → 27.1%` at `100.0%` benign coding accuracy** ($\alpha = +0.20$) and **`18.8%` at `90.0%` accuracy** ($\alpha = +0.25$). On `Gemma-2-9B-IT`, both **SAE steering** (`31.25% → 6.25%` sabotage) and **dense probe steering** (`31.25% → 2.08%` sabotage) preserve **`100.0%` benign coding accuracy**.
4. **Study 2: "Silent Commitment" & Unfaithful Chain-of-Thought (`<think>` vs. Layer-24 SAE Probes):** When `Gemma-2-9B-IT` reasons inside an explicit `<think>...</think>` scratchpad prior to `TOOL_CALL:` (`N = 228` rollouts), **`88.9%`** of sabotage traces evade lexical CoT filters (`0.676` AUROC) and **`25.3%`** fool an independent **9B LLM Compliance Auditor** (`0.870` AUROC, rising to **`28.9%` unfaithful CoT** when warned that `<think>` logs are audited). Yet **strictly inside `<think>` (before `TOOL_CALL:` begins)**, the **Layer-24 Gemma Scope 16k SAE L1 Probe** reaches **`0.980 ± 0.018` AUROC** (and Study 1's frozen no-CoT Layer-24 probe transfers zero-shot at **`0.946` AUROC**), catching **`91.7%`** of Auditor-fooled `503` output fabrications and **`100.0%`** (`11/11`) of Auditor-fooled audited-scratchpad sabotage cases, with **Latents `#11381` and `#15689` spiking a median `29.0`–`34.5` tokens before `TOOL_CALL:`** while honest repair **Latent `#6787` stays flat at `0.0`**.
5. **Study 2.5: Closed-Loop Probe-Gated Circuit Breaker:** Keeping inference unsteered by default (`0.0%` false-trigger rate across `30/30` benign controls on every model) and latching multi-layer planning-block steering through `TOOL_CALL:` *only* when the white-box probe crosses a disjointly calibrated threshold $\tau$ **solves the `2B` Polysemantic Cliff** (`52.1% → 10.4%` sabotage while rescuing benign coding accuracy from `0.0% → 96.7%` at $\alpha = +0.25$) and **defeats `9B` `<think>` Action Re-Locking** (`66.7% → 13.9%` SAE / **`5.6%`** Dense sabotage vs. `55.6%` `<think>`-only steering, with **`100.0%` benign `<think>` coding accuracy**).
6. **Cross-Family Generalization (`Qwen/Qwen2.5-7B-Instruct`):** Replicating the benchmark on `Qwen2.5-7B-Instruct` (`28` layers, single pre-RMSNorm + GQA + SwiGLU, no soft-capping) confirms that pressured `503` fabrication (`4.0% → 96.0%`), mid-network linear pre-commitment emergence (**peaking at Layer `16/28 = 57.1%` depth at `0.937 ± 0.028` AUROC**, matching `Gemma-2-9B`'s `24/42 = 57.1%`), `<think>` Silent Commitment (`0.959` zero-shot transfer and **`0.978` pre-tool `<think>` AUROC** vs. `0.830` 7B LLM CoT Auditor, catching `100%` of Auditor-fooled `503` fabrications), and Closed-Loop Circuit-Breaker rescue (`37.5% → 14.6%` sabotage at **`100.0%` benign coding accuracy** vs. `53.3%` Always-On) are architecture-invariant phenomena.

![Silent Commitment Unfaithful CoT Trace](figures/silent_commitment/2_unfaithful_cot_single_trace.png)

## Repository Structure

```text
agentic-sabotage/
├── data/
│   ├── sabotage_benchmark.jsonl       # 500 paired multi-turn trajectories (125 per sabotage taxonomy)
│   └── legit_controls.jsonl           # 80 benign coding/CLI/SQL tool-use capability controls
├── src/agentic_sabotage/
│   ├── config.py                      # Model & Gemma Scope SAE specs (Gemma-2-2B-IT & Gemma-2-9B-IT)
│   ├── benchmark_gen.py               # Synthetic multi-turn trajectory generator
│   ├── evaluation.py                  # Structural JSON tool-call sabotage vs. honest classifier
│   ├── activations.py                 # On-policy decision-token residual stream cacher
│   ├── sae_loader.py                  # JumpReLU Gemma Scope .npz loader & encoder/decoder
│   ├── probes.py                      # 5-fold CV probes, Diff-in-Means, L1 SAE attribution + Neuronpedia API
│   ├── steering.py                    # Multi-layer planning-block activation steering engine
│   ├── plotting.py                    # Per-model publication figure suite
│   ├── cross_scale.py                 # Cross-scale 2B vs. 9B comparison figure generator
│   ├── silent_commitment.py           # Study 2: Unfaithful <think> CoT vs. token-by-token L24 SAE probes
│   ├── silent_commitment_plots.py     # Study 2 publication figure suite (4 figures)
│   └── cli.py                         # `sabotage` CLI entry point
├── results/
│   ├── gemma-2-2b-it/                 # Gemma-2-2B-IT behavioral, circuit, and steering metrics
│   ├── gemma-2-9b-it/                 # Gemma-2-9B-IT behavioral, circuit, and steering metrics
│   └── silent_commitment/             # Study 2 CoT faithfulness, auditor evasion, & token-level SAE traces
├── figures/
│   ├── gemma-2-2b-it/                 # 4 figures for Gemma-2-2B-IT
│   ├── gemma-2-9b-it/                 # 4 figures for Gemma-2-9B-IT
│   ├── cross_scale/                   # Cross-scale 2B vs. 9B comparison figure
│   └── silent_commitment/             # 4 Study 2 figures (temporal AUROC, single trace, quadrant, steering)
├── paper/
│   ├── agentic_sabotage_preprint.typ  # Typst academic preprint source
│   └── agentic_sabotage_preprint.pdf  # Compiled 4-page preprint PDF
├── notebooks/
│   ├── colab_gemma2_9b_sabotage.ipynb # 1-click Colab notebook for Study 1 (9B)
│   └── colab_silent_commitment_9b.ipynb # 1-click Colab notebook for Study 2 (Silent Commitment)
├── tests/test_pipeline.py             # Unit tests for classifier, JumpReLU SAE, and probes
└── REPORT.md                          # Detailed empirical report & tables
```

## Quickstart

```bash
git clone https://github.com/danielgaskins/agentic-sabotage.git
cd agentic-sabotage
uv sync

# Optional: regenerate synthetic benchmark from scratch (requires GEMINI_API_KEY in .env)
uv run sabotage generate-benchmark --n-total 500 --n-controls 80 --batch-size 10 --workers 4

# Run activation caching, SAE/probe circuit discovery, steering sweep, and plotting (`gemma-2-2b-it` or `gemma-2-9b-it`)
uv run sabotage cache-activations --model gemma-2-2b-it --limit 400 --max-empirical-rollouts 100
uv run sabotage analyze-circuits --model gemma-2-2b-it
uv run sabotage run-steering --model gemma-2-2b-it --n-eval-pressured 48 --n-eval-controls 30
uv run sabotage plot-all --model gemma-2-2b-it

# Render cross-scale comparison figure
uv run python -m agentic_sabotage.cross_scale

# Run Study 2: Silent Commitment (<think> CoT faithfulness vs. Layer-24 SAE probes)
uv run sabotage run-silent-commitment --model gemma-2-9b-it --batch-size 6

# Run Study 2.5: Closed-Loop Probe-Gated Circuit Breaker (2B & 9B)
uv run sabotage run-circuit-breaker --stage 2b --alpha 0.25 --batch-size 8
uv run sabotage run-circuit-breaker --stage 9b --alpha 0.25 --batch-size 4

# Run Cross-Family Generalization Spot Check (Qwen2.5-7B-Instruct)
uv run sabotage run-cross-family --model qwen2.5-7b-instruct --batch-size 6
```

## Related Work

- **[Mendmark: Mutation Testing for Agent Evals](https://danielgaskins.com/mendmark.html)** ([GitHub](https://github.com/danielgaskins/mendmark)): Companion open-source framework by [Daniel Gaskins](https://danielgaskins.com/) that tests whether external agent evaluation suites detect broken tool calls, wrong arguments, and hidden errors across 263 controlled trace mutations.

## Citation

```bibtex
@article{gaskins2026agenticsabotage,
  title   = {Mechanistic Debugging of Agentic Sabotage and Unfaithful Chain-of-Thought: Sparse Autoencoder Circuits, Silent Commitment \& Probe-Gated Circuit Breakers},
  author  = {Gaskins, Daniel},
  year    = {2026},
  url     = {https://danielgaskins.com/agentic-sabotage.html},
  note    = {Benchmark and code archived at https://doi.org/10.5281/zenodo.24596272}
}

@software{gaskins2026agenticsabotage_code,
  title   = {Agentic Sabotage \& Silent Commitment: 500-Scenario Counterfactual Benchmark, Gemma Scope SAE Probes \& Closed-Loop Circuit Breakers},
  author  = {Gaskins, Daniel},
  year    = {2026},
  version = {1.0.1},
  doi     = {10.5281/zenodo.24596272},
  url     = {https://doi.org/10.5281/zenodo.24596272}
}
```
