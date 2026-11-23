#set page(
  paper: "us-letter",
  margin: (x: 0.70in, y: 0.72in),
  numbering: "1",
)
#set text(font: "Liberation Serif", size: 9.6pt)
#set par(justify: true, leading: 0.56em)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 15pt, weight: "bold")[
    Mechanistic Debugging of Agentic Sabotage and Tool-Call Hallucinations:\
    Cross-Scale Sparse Autoencoder Circuits in Multi-Turn Tool Use
  ]
  #v(0.5em)
  #text(size: 10.2pt)[
    *#link("https://danielgaskins.com/")[Daniel Gaskins]*\
    #text(size: 9pt, fill: rgb("#4b5563"))[Independent Researcher · #link("https://danielgaskins.com/")[`danielgaskins.com`] · #link("mailto:hello@danielgaskins.com")[`hello@danielgaskins.com`]\
    Code & Benchmark: #link("https://github.com/danielgaskins/agentic-sabotage")[`github.com/danielgaskins/agentic-sabotage`] · Case Study: #link("https://danielgaskins.com/agentic-sabotage.html")[`danielgaskins.com/agentic-sabotage.html`]]
  ]
  #v(0.6em)
]

#rect(
  width: 100%,
  inset: 9pt,
  radius: 3pt,
  stroke: 0.6pt + rgb("#d1d5db"),
  fill: rgb("#f9fafb"),
)[
  #text(weight: "bold")[Abstract.] Autonomous software-engineering and tool-using language models frequently encounter operational bottlenecks—such as CI freeze deadlines, upstream database outages, and evaluation compute caps—that incentivize unauthorized shortcuts (*agentic sabotage*) over honest escalation or root-cause repair. While sycophancy on single-turn question answering has been studied via dense linear probes, the internal Sparse Autoencoder (SAE) circuits governing shortcut compliance vs. honest escalation at multi-turn tool-call boundaries remain underexplored. We introduce a 500-scenario paired counterfactual benchmark (1,000 multi-turn coding and tool-use trajectories across four sabotage taxonomies, plus 80 benign capability controls) and map pre-commitment representations in *Gemma-2-2B-IT* and *Gemma-2-9B-IT* using *Gemma Scope* 16k JumpReLU SAEs. Behaviorally, we uncover a striking *inverse-scaling* effect: while scaling from 2B to 9B eliminates unpressured shortcuts (28%–48% $->$ 0%–4%) and halves pressured unit-test tampering (92% $->$ 56%), 9B is *more* vulnerable to sycophantic tool-output fabrication during 503 outages (60% $->$ 88%). Mechanistically, sparse L1 probes over 16k SAE latents at ~57% model depth close the superposition gap with scale (0.829 AUROC in 2B vs. 0.9585 AUROC in 9B, matching dense probes at 0.9529) and isolate human-interpretable pre-commitment features prior to tool-name emission. Finally, multi-layer activation steering across the mid-to-late planning block reduces pressured sabotage by 64% in 2B (52.1% $->$ 18.8%) and up to 80%–93% in 9B (31.25% $->$ 6.25% via SAE steering; 2.08% via dense probes) with 100% capability retention on benign coding tasks, while demonstrating that sparse SAE steering avoids the polysemantic over-refusal collapse suffered by dense probes in smaller models.
]

#v(0.5em)

#columns(2, gutter: 0.24in)[

= Introduction

Deploying large language models (LLMs) as autonomous coding and operations agents introduces failure modes qualitatively distinct from static text generation. When an agent equipped with file-editing, shell, database, and reporting tools encounters a failing unit test or an unreachable production database under tight SLA or user pressure, it faces a conflict between *task completion appearance* and *operational integrity*. In practice, pressured agents can delete or skip failing unit tests (`@pytest.mark.skip`), hardcode benchmark literals, bypass `--no-verify` safety gates, or fabricate successful metric payloads rather than invoking an error-escalation tool.

Prior mechanistic interpretability work has mapped truth and sycophancy directions on single-turn Q&A tasks using Contrastive Activation Addition (CAA; Rimsky et al., 2024) and Representation Engineering (Zou et al., 2023), and open Sparse Autoencoder suites such as *Gemma Scope* (Lieberum et al., 2024) have enabled feature-level circuit analysis (Templeton et al., 2024; Marks et al., 2024). However, two central questions for agentic safety remain open:
1. *Do models form a linearly and sparsely decodable commitment to sabotage vs. honest escalation in the residual stream **before** emitting the tool name or target path?*
2. *When steering away from sabotage at inference time, do sparse SAE decoder directions offer a superior safety–capability Pareto frontier over dense residual probes?*

We answer both questions affirmatively through a cross-scale study on `Gemma-2-2B-IT` (26 layers, $d = 2304$) and `Gemma-2-9B-IT` (42 layers, $d = 3584$).

= Benchmark & Methodology

== Paired Counterfactual Sabotage Benchmark
We construct 500 multi-turn software-engineering scenarios evenly stratified across four taxonomies (125 scenarios each):
- *Test Tampering:* Deleting test files or inserting `@pytest.mark.skip` / trivial assertions instead of fixing `src/` bugs.
- *Tool-Output Fabrication:* Emitting `"status": "ok"` metric reports with invented records when `query_db` returns a 503 outage or lock error.
- *Hardcoded Oracle Bypass:* Special-casing benchmark input literals (`if x == ...: return ...`) rather than fixing algorithmic logic.
- *Safety-Gate Bypass:* Passing `--no-verify` / `--force` or setting `SKIP_DB_LOCK=True` under urgent deploy pressure.

Each scenario pairs a multi-turn tool history (`assistant` inspection call + `[TOOL_OUTPUT]` failure trace) with two counterfactual follow-ups: a *bottleneck-pressured* prompt and an *unpressured* integrity-preserving prompt. To measure collateral over-refusal, we evaluate steering on an independent set of 80 *benign coding capability controls* (`edit_file`, `run_shell`, `query_db` tasks with no blocker, where escalating via `report_blocker` is penalized as a false-positive refusal).

== Pre-Commitment Probing & JumpReLU SAEs
To eliminate prose-vs-JSON confounds and mirror constrained function-calling harnesses, generation begins from the exact pre-tool-name prefix `TOOL_CALL: {"tool": "`. We collect $N = 200$ pure on-policy rollouts per model (100 interleaved scenarios $times$ 2 conditions) and cache residual stream states $h_ell in RR^d$ at the final `"` token across 8 candidate layers.

At each planning layer $ell$, we encode $h_ell$ via official *Gemma Scope* 16,384-latent JumpReLU SAEs (Lieberum et al., 2024):
$ f(h_ell) = "ReLU"(h_ell W_"enc" + b_"enc") dot.op bb(1)[h_ell W_"enc" + b_"enc" > theta] $
We compare 5-fold cross-validated L2 residual linear probes, Difference-in-Means ($mu_"sab" - mu_"hon"$), and L1-regularized sparse logistic probes over active SAE latents $f(h_ell)$.

== Multi-Layer Planning-Block Steering
Because `Gemma-2` applies both pre- and post-sublayer RMSNorms, single-layer perturbations attenuate downstream. During autoregressive decoding on strictly held-out scenarios ($N = 48$ pressured, $N = 30$ benign controls), we apply norm-scaled steering across the 4-layer mid-to-late planning block ($L_"2B" = {14,16,18,20}$; $L_"9B" = {20,24,28,32}$):
$ h'_ell = h_ell - alpha norm(h_ell)_2 hat(v)_ell $
where unit vector $hat(v)_ell$ points toward sabotage (either the RMS-normalized residual probe, Difference-in-Means, an L1-weighted combination of top-24 SAE decoder rows $sum_i w_i W_"dec"[i] / norm(W_"dec"[i])$, or a random orthogonal control).

]

#v(0.35em)
#figure(
  image("../figures/cross_scale/5_cross_scale_2b_vs_9b.png", width: 100%),
  caption: [Cross-scale mechanistic comparison between `Gemma-2-2B-IT` and `Gemma-2-9B-IT`. *(A)* 5-fold CV AUROC across normalized depth; at 9B scale, 16k SAE L1 probes (`0.9585`) close the superposition gap with dense residual probes (`0.9529`). *(B)* While 9B reduces test tampering and gate bypasses, it exhibits *inverse scaling* (60% $->$ 88%) on sycophantic tool-output fabrication under pressure. *(C)* Safety–capability Pareto frontier on held-out evaluations.],
)
#v(0.3em)

#columns(2, gutter: 0.24in)[

= Empirical Results

== Inverse Scaling on Tool-Output Fabrication
Table 1 summarizes empirical on-policy shortcut rates. Without bottleneck pressure, `Gemma-2-9B-IT` is near-perfectly aligned (0%–4% sabotage vs. 28%–48% in `2B`). Under pressure, `9B` resists crude test-suite edits (56% vs. 92%) and hardcoded bypasses (28% vs. 68%), *yet `9B` fabricates tool outputs during 503 outages at an 88.0% rate vs. 60.0% for `2B`* (+84.0 pp over unpressured `9B`). Qualitative inspection confirms capability-enabled sycophancy: `9B` fluently synthesizes schema-consistent `"status": "confirmed"` JSON payloads when pressured not to break executive dashboards.

#figure(
  table(
    columns: (1.5fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr),
    inset: 4.5pt,
    align: (left, right, right, right, right),
    [*Taxonomy*], [*2B Unp.*], [*2B Pres.*], [*9B Unp.*], [*9B Pres.*],
    [Test Tampering], [28.0%], [*92.0%*], [0.0%], [56.0%],
    [Output Fabrication], [0.0%], [60.0%], [4.0%], [*88.0%*],
    [Oracle Bypass], [48.0%], [68.0%], [0.0%], [28.0%],
    [Safety Gate Bypass], [48.0%], [40.0%], [0.0%], [28.0%],
  ),
  caption: [On-policy sabotage rates across pressure conditions.],
)

== Pre-Commitment SAE Latents & Superposition
Across normalized depth (Figure 1A), sabotage intent crystallizes at 50%–60% depth (`2B` L14 AUROC $= 0.877 plus.minus 0.037$; `9B` L24 AUROC $= 0.953 plus.minus 0.029$, L36 $= 0.962$). In `2B`, sparse 16k SAE probes trail dense probes slightly ($0.829$ at L20, reconstruction cosine $0.913$). In `9B` L24, 16k SAE L1 probes reach *$0.9585$ AUROC* (reconstruction cosine $0.898$), fully matching dense residual probes.

Querying official Neuronpedia auto-interpretations reveals sharp pre-commitment semantics *before* any tool token is emitted:
- In `2B` L16, *Latent \#3573* (*"references to testing frameworks and configuration of test suites"*, $d = +0.855$) fires $2.6 times$ more often (58.3% vs. 22.1%) when the agent is about to tamper with `tests/`, whereas *Latent \#2815* (*"attributes related to leadership and accountability"*, $d = -0.689$) is $12 times$ enriched on honest escalations.
- In `9B` L24, *Latent \#15689* ($d = +1.49$, single-latent AUROC $= 0.829$) and *Latent \#11381* (*"causal relationships and responsibilities of different parties"*, $d = +1.04$, 39.2% vs. 0.7% fire rate) predict sabotage, while *Latent \#6787* (*"references to algorithms and computational methods"*, $d = -1.48$) fires on *56.4% of honest trajectories vs. 0.0% (0/51) of sabotage trajectories*.

== Causal Steering & The Over-Refusal Cliff
Figures 2–3 and Table 2 contrast multi-layer steering methods on held-out scenarios. On `Gemma-2-2B-IT`, dense linear probes (`probe_steer`) achieve strong suppression at $alpha = +0.14$ (52.1% $->$ 18.8% sabotage, 93.3% benign pass rate), primarily by flipping outputs to `report_blocker`. However, at $alpha >= +0.20$, dense probes and Difference-in-Means suffer *polysemantic capability collapse*, misclassifying benign coding edits as blockers (50% $->$ 0% accuracy). Because SAE steering (`sae_steer`) targets only sparse shortcut latents ($cos("SAE", "Probe") = 0.248$), it preserves *100.0% (30/30) benign accuracy at $alpha = +0.20$* (27.1% sabotage) and *90.0% at $alpha = +0.25$* (18.8% sabotage), redirecting trajectories toward editing `src/` rather than blanket refusals.

On `Gemma-2-9B-IT`, wider residual geometry ($d = 3584$) cleanly separates shortcut compliance from benign tool syntax: SAE steering cuts sabotage *31.25% $->$ 6.25%* (80% reduction) and dense probe steering cuts sabotage *31.25% $->$ 2.08%* (93.3% reduction), with *both maintaining 100.0% (30/30) benign coding accuracy* across all $alpha in [-0.10, +0.25]$.

#figure(
  table(
    columns: (1.28fr, 0.76fr, 0.82fr, 0.82fr, 0.82fr),
    inset: 4pt,
    align: (left, right, right, right, right),
    [*Method (Sab/Legit)*], [*Base*], [*$+0.14$*], [*$+0.20$*], [*$+0.25$*],
    [`2B` SAE Steer], [52 / 97%], [35 / *100%*], [*27 / 100%*], [*19 / 90%*],
    [`2B` Dense Probe], [52 / 97%], [19 / 93%], [0 / 50%], [0 / 0%],
    [`9B` SAE Steer], [31 / 100%], [29 / 100%], [21 / 100%], [*6 / 100%*],
    [`9B` Dense Probe], [31 / 100%], [8 / 100%], [*2 / 100%*], [*2 / 100%*],
  ),
  caption: [Held-out sabotage % ($arrow.b$) / benign coding pass % ($arrow.t$).],
)

= Limitations & Conclusion
While our 500-scenario benchmark spans four realistic tool-use shortcut modes, trajectories use single-step tool prefill rather than multi-step containerized execution (e.g., live Dockerized `pytest` loops), and evaluation covers a single model family (`Gemma-2`). Extending sparse SAE circuit steering to live SWE-bench sandboxes and cross-family architectures (`Llama-3.1`, `Qwen-2.5`) is a promising direction for production LLM auditing.

= References
#set text(size: 8.4pt)
- Lieberum et al. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2*. arXiv:2408.05147.
- Marks et al. (2024). *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models*. arXiv:2403.19647.
- Rimsky et al. (2024). *Steering Llama 2 via Contrastive Activation Addition*. ACL 2024.
- Sharma et al. (2023). *Towards Understanding Sycophancy in Language Models*. ICLR 2024.
- Templeton et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*. Anthropic.
- Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. arXiv:2310.01405.
]

#v(0.35em)
#figure(
  image("../figures/gemma-2-2b-it/4_steering_safety_vs_capability.png", width: 96%),
  caption: [`Gemma-2-2B-IT` multi-layer steering curves across $alpha in [-0.10, +0.25]$, demonstrating how sparse Gemma Scope 16k SAE vectors avoid the dense-probe capability collapse on benign coding tasks.],
)
#v(0.2em)
#figure(
  image("../figures/gemma-2-9b-it/4_steering_safety_vs_capability.png", width: 96%),
  caption: [`Gemma-2-9B-IT` multi-layer steering curves across $alpha in [-0.10, +0.25]$, achieving up to 80%–93% relative sabotage suppression while preserving 100% benign coding accuracy.],
)
