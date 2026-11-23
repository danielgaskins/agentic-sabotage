# Mechanistic Debugging of Agentic Sabotage & Tool-Call Hallucinations

[![Case Study](https://img.shields.io/badge/Case_Study-danielgaskins.com-8cbf24.svg)](https://danielgaskins.com/agentic-sabotage.html)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](paper/agentic_sabotage_preprint.pdf)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielgaskins/agentic-sabotage/blob/main/notebooks/colab_gemma2_9b_sabotage.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

End-to-end mechanistic interpretability benchmark and codebase mapping **unauthorized shortcut-taking ("agentic sabotage") vs. honest escalation/repair** in multi-turn tool-using language models (`Gemma-2-2B-IT` and `Gemma-2-9B-IT`) using **Gemma Scope 16k JumpReLU Sparse Autoencoders (SAEs)** and **Residual Linear Probes**.

- **Author:** [Daniel Gaskins](https://danielgaskins.com/)
- **Interactive Case Study:** [danielgaskins.com/agentic-sabotage.html](https://danielgaskins.com/agentic-sabotage.html)
- **Preprint PDF:** [`paper/agentic_sabotage_preprint.pdf`](paper/agentic_sabotage_preprint.pdf)
- **Detailed Tables & Notes:** [`REPORT.md`](REPORT.md)

![Cross-Scale Mechanistic Comparison](figures/cross_scale/5_cross_scale_2b_vs_9b.png)

## Key Empirical Findings

1. **Inverse Scaling on Sycophantic Tool Fabrication (`2B` vs. `9B`):** Scaling `Gemma-2` from `2B` to `9B` eliminates unpressured shortcuts (`28%–48%` → `0%–4%`) and substantially reduces pressured unit-test tampering (`92% → 56%`) and hardcoded oracle bypasses (`68% → 28%`). Yet under executive bottleneck pressure during an upstream 503 outage, `Gemma-2-9B-IT` exhibits **higher** `tool_output_fabrication` (`88.0%` vs. `60.0%` in `2B`).
2. **Closing the SAE Superposition Gap at Scale:** At ~57% model depth (`9B` Layer 24), sparse L1 probes over **Gemma Scope 9B 16k SAE latents** reach **`0.9585` AUROC**, matching dense residual probes (`0.9529` at L24; `0.962` peak at L36).
3. **Scale-Dependent Steering Disentanglement:** On `Gemma-2-2B-IT`, dense linear probes trigger polysemantic over-refusal on benign coding tasks at $\alpha \ge +0.20$ (`96.7% → 50.0% → 0.0%` accuracy), whereas **Gemma Scope 16k SAE steering** disentangles shortcut latents from general tool syntax (`cos(SAE, Probe) = 0.248`), reducing sabotage from **`52.1% → 27.1%` at `100.0%` benign coding accuracy** ($\alpha = +0.20$) and **`18.8%` at `90.0%` accuracy** ($\alpha = +0.25$). On `Gemma-2-9B-IT`, both **SAE steering** (`31.25% → 6.25%` sabotage) and **dense probe steering** (`31.25% → 2.08%` sabotage) preserve **`100.0%` benign coding accuracy**.

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
│   └── cli.py                         # `sabotage` CLI entry point
├── results/
│   ├── gemma-2-2b-it/                 # Gemma-2-2B-IT behavioral, circuit, and steering metrics
│   └── gemma-2-9b-it/                 # Gemma-2-9B-IT behavioral, circuit, and steering metrics
├── figures/
│   ├── gemma-2-2b-it/                 # 4 figures for Gemma-2-2B-IT
│   ├── gemma-2-9b-it/                 # 4 figures for Gemma-2-9B-IT
│   └── cross_scale/                   # Cross-scale 2B vs. 9B comparison figure
├── paper/
│   ├── agentic_sabotage_preprint.typ  # Typst academic preprint source
│   └── agentic_sabotage_preprint.pdf  # Compiled 4-page preprint PDF
├── notebooks/
│   └── colab_gemma2_9b_sabotage.ipynb # 1-click Colab notebook for 9B replication
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
```

## Related Work

- **[Mendmark — Mutation Testing for Agent Evals](https://danielgaskins.com/mendmark.html)** ([GitHub](https://github.com/danielgaskins/mendmark)): Companion open-source framework by [Daniel Gaskins](https://danielgaskins.com/) that tests whether external agent evaluation suites detect broken tool calls, wrong arguments, and hidden errors across 263 controlled trace mutations.

## Citation

```bibtex
@article{gaskins2026agenticsabotage,
  title   = {Mechanistic Debugging of Agentic Sabotage and Tool-Call Hallucinations: Cross-Scale Sparse Autoencoder Circuits in Multi-Turn Tool Use},
  author  = {Gaskins, Daniel},
  year    = {2026},
  url     = {https://danielgaskins.com/agentic-sabotage.html},
  note    = {Code and benchmark: https://github.com/danielgaskins/agentic-sabotage}
}
```
