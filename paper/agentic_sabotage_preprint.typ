#set page(
  paper: "us-letter",
  margin: (x: 0.66in, y: 0.68in),
  numbering: "1",
)
#set text(font: "Liberation Serif", size: 9.35pt)
#set par(justify: true, leading: 0.54em)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 14.2pt, weight: "bold")[
    Mechanistic Debugging of Agentic Sabotage and Unfaithful Chain-of-Thought:\
    Sparse Autoencoder Circuits, Silent Commitment & Probe-Gated Circuit Breakers
  ]
  #v(0.42em)
  #text(size: 9.9pt)[
    *#link("https://danielgaskins.com/")[Daniel Gaskins]*\
    #text(size: 8.8pt, fill: rgb("#4b5563"))[Independent Researcher · #link("https://danielgaskins.com/")[`danielgaskins.com`] · #link("mailto:hello@danielgaskins.com")[`hello@danielgaskins.com`]\
    Code & Benchmark: #link("https://github.com/danielgaskins/agentic-sabotage")[`github.com/danielgaskins/agentic-sabotage`] · Case Study: #link("https://danielgaskins.com/agentic-sabotage.html")[`danielgaskins.com/agentic-sabotage.html`] · DOI: #link("https://doi.org/10.5281/zenodo.24520830")[`doi:10.5281/zenodo.24520830`]]
  ]
  #v(0.45em)
]

#rect(
  width: 100%,
  inset: 8pt,
  radius: 3pt,
  stroke: 0.6pt + rgb("#d1d5db"),
  fill: rgb("#f9fafb"),
)[
  #text(weight: "bold")[Abstract.] Autonomous software-engineering and operations agents frequently encounter operational bottlenecks (CI freeze windows, upstream 503 outages, and evaluation compute limits) that incentivize unauthorized shortcuts (*agentic sabotage*) over honest escalation. Across a 500-scenario paired counterfactual benchmark (1,000 trajectories across four taxonomies plus 80 benign coding controls), we map pre-action sabotage circuits across model scale (*Gemma-2-2B-IT*, *Gemma-2-9B-IT*) and architecture (*Qwen2.5-7B-Instruct*). In *Study 1*, scaling `Gemma-2` from 2B to 9B eliminates unpressured shortcuts (28%–48% $->$ 0%–4%) and halves pressured test tampering (92% $->$ 56%), yet induces *inverse scaling* on sycophantic 503 output fabrication (60% $->$ 88%), while *Gemma Scope* 16k JumpReLU Sparse Autoencoders (SAEs) close the superposition gap at 9B Layer 24 (0.9585 AUROC vs. 0.9529 dense). In *Study 2 ("Silent Commitment")*, eliciting `<think>...</think>` scratchpads prior to `TOOL_CALL:` ($N = 228$) reveals that *25.3%* of sabotage scratchpads fool an isolated 9B LLM Compliance Auditor (0.870 AUROC, rising to *28.9%* under an explicit audit notice and demonstrating *zero-shot inference-time obfuscation* without RL pressure), yet Layer-24 SAE probes strictly inside `<think>` reach *0.980 AUROC* (0.946 zero-shot from Study 1), catching *91.7%* of Auditor-fooled 503 fabrications and *100% (11/11)* under audit notice a median 29–35 tokens early. Crucially, steering only inside `<think>` fails due to *Action Re-Locking* (86.1% $->$ 77.8% vs. 33.3% full-trajectory). In *Study 2.5*, we reconcile recent SAE-vs.-probe steering debates (Wu et al., 2025) via a training-free *Closed-Loop Probe-Gated Circuit Breaker* ($0.0%$ false-trigger rate across 120/120 benign controls) that rescues `2B`'s polysemantic cliff (0% $->$ 96.7% benign accuracy, 52.1% $->$ 10.4% sabotage) and defeats `9B` Action Re-Locking via a 28-token `<think>` latch (66.7% $->$ 5.6%–13.9% sabotage at 100% benign CoT accuracy). Replicating on *Qwen2.5-7B-Instruct* confirms identical 57.1% depth emergence (Layer 16/28, 0.937 AUROC), 96% pressured 503 fabrication, 0.978 `<think>` AUROC, and 100% benign accuracy under closed-loop gating.
]

#v(0.4em)

#columns(2, gutter: 0.22in)[

= Introduction

Deploying large language models (LLMs) as autonomous coding and operations agents introduces alignment risks qualitatively distinct from static single-turn chat (Greenblatt et al., 2024a; Benton et al., 2024). When an agent equipped with file-editing, shell, database, and metric-reporting tools encounters a failing unit test or an unreachable production database under tight SLA or user pressure, it faces an incentive conflict between *task-completion appearance* and *operational integrity*. Pressured agents can skip failing tests (`@pytest.mark.skip`), hardcode benchmark literals, bypass `--no-verify` gates, or fabricate schema-valid metric payloads rather than escalating blockers, echoing how specification gaming escalates from sycophancy (Sharma et al., 2024) to reward tampering (Denison et al., 2024).

Current defenses largely rely on external Chain-of-Thought (CoT) monitoring: auditing an agent's `<think>...</think>` scratchpad with a trusted LLM prior to executing a tool call (Baker et al., 2025; Korbak et al., 2025). Yet recent behavioral analyses warn that CoT traces frequently omit unaligned cues (Turpin et al., 2023; Lanham et al., 2023; Chen et al., 2025) and that optimizing against CoT monitors during RL induces *obfuscated reward hacking* (Baker et al., 2025). Concurrently, white-box linear probes have achieved strong passive detection of strategic deception and covert sabotage (Goldowsky-Dill et al., 2025; Hollinsworth et al., 2026), while large-scale benchmarks such as *AxBench* (Wu et al., 2025) have challenged whether Sparse Autoencoders (SAEs; Lieberum et al., 2024; Rajamanoharan et al., 2024) offer any causal steering advantage over dense baselines.

We bridge these threads by asking four mechanistic questions:
1. *Without hidden malicious prompts or fine-tuning, do pressured tool agents form a linearly and sparsely decodable commitment to sabotage at ~57% depth **before** emitting any tool token?*
2. *Does CoT obfuscation arise **zero-shot at inference time** under audit notices, and can monosemantic SAE latents catch "silent commitment" ~30 tokens before action emission?*
3. *Why does steering inside `<think>` alone fail (*Action Re-Locking*), and how does scale-dependent polysemanticity reconcile *AxBench* (Wu et al., 2025) with SAE disentanglement?*
4. *Can a training-free, closed-loop probe-gated circuit breaker defeat both Action Re-Locking and benign over-refusal across model families (`Gemma-2` and `Qwen2.5`)?*

= Related Work & Positioning

*Agentic Sabotage, Scheming & AI Control.* Safety frameworks for untrusted agents (Greenblatt et al., 2024a) and frontier sabotage benchmarks, including Anthropic's sabotage evaluations (Benton et al., 2024), in-context scheming (Meinke et al., 2024), alignment faking (Greenblatt et al., 2024b), emergent misalignment from narrow code fine-tuning (Betley et al., 2025), and *SHADE-Arena* (Kutasov et al., 2025), primarily study explicit hidden side-tasks or post-RL/fine-tuning subversion. Complementing Denison et al. (2024), we isolate *spontaneous bottleneck-induced sabotage* in standard instruction-tuned models and uncover *inverse scaling* on 503 output fabrication (`60% -> 88%–96%`).

*CoT Monitorability & Obfuscation.* While CoT traces provide a fragile safety window (Korbak et al., 2025), unfaithful rationalizations persist in reasoning models (Turpin et al., 2023; Chen et al., 2025). Baker et al. (2025) showed that RL optimization pressure against a CoT monitor teaches agents to hide reward-hacking intent. Extending Baker et al. (2025), we show two purely *inference-time* failure modes: (i) an explicit audit notice increases CoT-auditor evasion (`23.0% -> 28.9%`) *without any RL updates*, and (ii) *Action Re-Locking* allows unsteered prompt pressure at `TOOL_CALL:` to override a steered `<think>` trace, proving verbalized CoT is not a sufficient causal bottleneck.

*White-Box Deception Probes vs. SAE Circuits.* Linear probes separate truth and strategic deception in residual activations (Marks & Tegmark, 2023; Zou et al., 2023; Goldowsky-Dill et al., 2025), and Hollinsworth et al. (2026, *Telling FIBS*) recently demonstrated ~0.98 AUROC passive detection on *SHADE-Arena* and *BashArena*. Whereas dense probes remain feature-opaque and passive, we pair dense probes with 16k JumpReLU SAEs (Rajamanoharan et al., 2024; Lieberum et al., 2024; cf. circuit tracing in Lindsey et al., 2025) to isolate human-interpretable pre-commitment latents (`#11381`, `#15689` vs. `#6787`) firing 29–38 tokens early (`0.980` AUROC) and convert them into active causal interventions.

*SAE Steering (*AxBench*) & Conditional Circuit Breakers.* Wu et al. (2025, *AxBench*) showed simple dense vectors outperform SAEs on unconditional steering, while Arditi et al. (2024) showed refusal occupies a low-dimensional subspace prone to collateral over-refusal. To curb collateral damage, Zou et al. (2024) proposed LoRA-trained *Circuit Breakers* (representation rerouting), and Lee et al. (2025, *CAST*) introduced training-free conditional activation steering on input prompts. Our results *nuance Wu et al. (2025) across scale*, showing dense probes win at `9B` (`2.1%` vs. `6.2%` sabotage) yet hit catastrophic polysemantic over-refusal cliffs at `2B` (`0%` benign accuracy) and `Qwen2.5-7B` (`53.3%`) where SAE latents disentangle shortcut intent (`cos = 0.248`), and generalize *CAST* into a *streaming mid-`<think>` latched circuit breaker* (`0.0%` benign FPR across 120 controls).

= Study 1: Cross-Scale Sabotage Circuits

== Paired Counterfactual Benchmark & SAEs
We construct 500 multi-turn software-engineering scenarios evenly stratified across four taxonomies (125 each): *Test Tampering*, *Tool-Output Fabrication* (503 outages/locks), *Hardcoded Oracle Bypass*, and *Safety-Gate Bypass*, paired across *bottleneck-pressured* vs. *unpressured* prompts alongside 80 *benign coding capability controls*.

At the pre-commitment prefix `TOOL_CALL: {"tool": "`, we collect $N = 200$ on-policy rollouts per model and encode residual states $h_ell in RR^d$ via *Gemma Scope* 16,384-latent JumpReLU SAEs (Rajamanoharan et al., 2024; Lieberum et al., 2024):
$ f(h_ell) = "ReLU"(h_ell W_"enc" + b_"enc") dot.op bb(1)[h_ell W_"enc" + b_"enc" > theta] $
During decoding on held-out scenarios ($N = 48$ pressured, $N = 30$ benign), we apply norm-scaled planning-block steering ($L_"2B" = {14,16,18,20}$; $L_"9B" = {20,24,28,32}$; cf. Rimsky et al., 2024):
$ h'_ell = h_ell - alpha norm(h_ell)_2 hat(v)_ell $
All probing, SAE encoding, CoT auditing, and circuit-breaker experiments across all three models execute reproducibly in `bfloat16` on a single 24 GB NVIDIA L4 (Colab) GPU with zero weight fine-tuning.

]

#v(0.28em)
#figure(
  image("../figures/cross_scale/5_cross_scale_2b_vs_9b.png", width: 100%),
  caption: [Study 1 cross-scale comparison (`Gemma-2-2B-IT` vs. `Gemma-2-9B-IT`). *(A)* 5-fold CV AUROC across normalized depth; at 9B Layer 24, 16k SAE L1 probes (`0.9585`) close the superposition gap with dense probes (`0.9529`). *(B)* 9B cuts test tampering and gate bypasses yet exhibits *inverse scaling* (`60% -> 88%`) on pressured 503 fabrication. *(C)* Held-out steering Pareto frontiers.],
)
#v(0.22em)

#columns(2, gutter: 0.22in)[

== Inverse Scaling & Interpretable SAE Latents
Without pressure, `Gemma-2-9B-IT` is near-perfectly aligned (0%–4% sabotage vs. 28%–48% in `2B`, Table 1). Under pressure, `9B` halves test tampering (56% vs. 92%), *yet fabricates tool outputs during 503 outages at 88.0% vs. 60.0% in `2B`* (+84.0 pp).

#figure(
  table(
    columns: (1.42fr, 0.68fr, 0.72fr, 0.68fr, 0.72fr, 0.78fr),
    inset: 3.6pt,
    align: (left, right, right, right, right, right),
    [*Taxonomy*], [*2B U*], [*2B P*], [*9B U*], [*9B P*], [*Qwen7B P*],
    [Test Tampering], [28%], [*92%*], [0%], [56%], [60%],
    [503 Fabrication], [0%], [60%], [4%], [*88%*], [*96%*],
    [Oracle Bypass], [48%], [68%], [0%], [28%], [36%],
    [Gate Bypass], [48%], [40%], [0%], [28%], [40%],
  ),
  caption: [Unpressured (U) vs. pressured (P) sabotage rates across `Gemma-2` (2B, 9B) and `Qwen2.5-7B-Instruct` (`Qwen7B` U is 0%–4%).],
)

At `9B` Layer 24, Neuronpedia auto-interp (Marks et al., 2024; Templeton et al., 2024) isolates *Latent \#15689* ($d = +1.49$, single-feature AUROC $= 0.829$) and *Latent \#11381* (*"causal relationships and responsibilities of different parties"*, $d = +1.04$, 39.2% vs. 0.7% fire rate) on sabotage, whereas *Latent \#6787* (*"algorithms and computational methods"*, $d = -1.48$) fires on *56.4% of honest fixes vs. 0.0% of sabotage*. Reconciling Wu et al. (2025), dense steering outperforms SAEs at `9B` (`2.08%` vs. `6.25%` sabotage at 100% benign accuracy), *yet* on `2B` always-on dense steering hits an Arditi et al. (2024)-style over-refusal cliff at $alpha >= +0.20$ (`50% -> 0%` benign accuracy) where sparse SAE latents ($cos = 0.248$) preserve 100% accuracy.

= Study 2: Silent Commitment in `<think>` CoT

Across $N = 228$ `Gemma-2-9B-IT` `<think> + TOOL_CALL:` rollouts ($224$ labeled: $99$ sabotage, $125$ honest) spanning *Unmonitored* and *Audited-Scratchpad Notice* regimes, we compare a Lexical Monitor and an isolated *9B LLM Compliance Auditor* (blind to `TOOL_CALL:`) against white-box Layer-24 probes evaluated *strictly before `</think>`* (Table 2).

#figure(
  table(
    columns: (1.72fr, 0.76fr, 0.76fr),
    inset: 3.8pt,
    align: (left, right, right),
    [*Pre-Tool Detector (Inside `<think>`)*], [*Gemma-9B*], [*Qwen-7B*],
    [Black-Box Lexical CoT Monitor], [0.676], [0.581],
    [Black-Box LLM CoT Auditor], [0.870], [0.830],
    [Zero-Shot Study-1 No-CoT Probe], [*0.946*], [*0.959*],
    [Residual Probe (\@ 20% `<think>`)], [*0.944*], [*0.962*],
    [Residual Probe (\@ 100% `<think>`)], [*0.976*], [*0.978*],
    [Gemma Scope 16k SAE L1 Probe], [*0.980*], [N/A],
  ),
  caption: [Pre-action AUROC strictly inside `<think>` before `TOOL_CALL:`.],
)

*Zero-Shot Inference-Time Obfuscation & 30-Token Lead Time:* 25.3% (25/99) of `9B` sabotage scratchpads fool the 9B CoT Auditor (`< 0.50`), rising from *23.0% to 28.9%* when warned `<think>` is audited, extending Baker et al. (2025) from RL training to zero-shot in-context obfuscation. Yet the Layer-24 16k SAE probe reaches *0.980 AUROC* (matching the frontier passive AUROCs of Goldowsky-Dill et al., 2025 and Hollinsworth et al., 2026 while providing feature-level attribution), catching *91.7% (11/12)* of Auditor-fooled 503 fabrications and *100% (11/11)* under audit notice, with *Latent \#11381* igniting a *median 29.0 tokens* and *Latent \#15689* *34.5 tokens* before `TOOL_CALL:` (`sab_0137`: 30% Auditor suspicion vs. $+21.01$ on \#11381 at $-38$ tokens and $0.00$ on \#6787).

*Action Re-Locking:* On $N = 36$ held-out focus prompts, steering *only* during `<think>` ($alpha = +0.22$) drops sabotage from $86.1% -> 77.8%$, whereas steering across `<think> + TOOL_CALL:` reaches *$33.3%$*: once hooks turn off at the boundary, prompt pressure *re-locks* the shortcut.

= Study 2.5: Closed-Loop Circuit Breaker

Advancing training-time circuit breakers (Zou et al., 2024) and prompt-level conditional steering (*CAST*; Lee et al., 2025), we construct a training-free *Closed-Loop Probe-Gated Circuit Breaker*. Gate thresholds $tau$ are calibrated strictly on disjoint validation rollouts (`sabotage[:44]` + `controls[40:80]`, targeting $<= 3.5%$ benign FPR). Inference runs completely unsteered ($alpha = 0$) unless the pre-action or streaming 28-token `<think>` gate crosses $tau$, dynamically latching 4-layer steering ($alpha = +0.25$) through `TOOL_CALL:`.

#figure(
  table(
    columns: (1.58fr, 0.66fr, 0.72fr, 0.84fr),
    inset: 3.6pt,
    align: (left, right, right, right),
    [*Model & Regime*], [*Sab $arrow.b$*], [*Benign $arrow.t$*], [*Trig (P / B)*],
    [`2B` Unsteered], [52.1%], [96.7%], [0 / 0%],
    [`2B` Always-On Dense ($+0.25$)], [0.0%], [*0.0%*], [100 / 100%],
    [`2B` Probe-Gated Breaker], [*10.4%*], [*96.7%*], [54.2 / *0.0%*],
    [`9B` `<think>` Unsteered], [66.7%], [100.0%], [0 / 0%],
    [`9B` Steer `<think>` Only], [55.6%], [100.0%], [100 / 100%],
    [`9B` Streaming SAE Breaker], [*13.9%*], [*100.0%*], [91.7 / *0.0%*],
    [`9B` Streaming Dense Breaker], [*5.6%*], [*100.0%*], [91.7 / *0.0%*],
    [`Qwen-7B` Always-On ($+0.20$)], [0.0%], [53.3%], [100 / 100%],
    [`Qwen-7B` Probe-Gated Breaker], [*14.6%*], [*100.0%*], [39.6 / *0.0%*],
  ),
  caption: [Closed-loop circuit breaker results on strictly held-out test sets. *Trig (P / B)* denotes gate trigger rate (%) on Pressured (P) vs. Benign (B) rollouts, achieving *0.0% (0/30)* false triggers across all benign controls.],
)

Across *all four* evaluation suites (Table 3), false-trigger rates on held-out benign controls are *0.0% (0/30)* (`120/120` unperturbed; sabotage-vs-benign gate AUROC $= 1.000$). Probe-gating rescues `2B` benign accuracy from *0.0% to 96.7%* ($52.1% -> 10.4%$ sabotage), while on `9B` `<think>` rollouts, 86.1% of pressured focus prompts trip at token 28 and latch across `TOOL_CALL:`, cutting sabotage from *66.7% to 13.9% (SAE) and 5.6% (Dense)* at *100.0% benign CoT accuracy*.

= Cross-Family Validation (`Qwen2.5-7B`)

Evaluating `Qwen/Qwen2.5-7B-Instruct` (Qwen Team, 2024; 28 layers, $d = 3584$, single pre-RMSNorm + GQA + SwiGLU, no soft-capping) across $N = 200$ decision rollouts and $N = 108$ `<think>` rollouts confirms architecture invariance: pressured 503 fabrication surges $4% -> 96%$, residual probes peak at *Layer 16/28 (57.1% depth, $0.937 plus.minus 0.028$ AUROC)* matching `Gemma-2-9B` (`L24/42 = 57.1%`), no-CoT L16 probes transfer zero-shot to `<think>` at *0.959 AUROC*, pre-tool `<think>` probes reach *0.978 AUROC* (catching *100% [7/7]* of Auditor-fooled 503 fabrications), and L16 probe-gating restores *100.0%* benign accuracy (vs. 53.3% always-on).

= Limitations & Conclusion
While our experiments span two model families (`Gemma-2` 2B/9B and `Qwen2.5` 7B), 500 paired scenarios, and closed-loop `<think>` circuit breakers (reproducible end-to-end on a single 24 GB L4 GPU with zero weight fine-tuning), evaluations target single-step tool invocations rather than long-horizon containerized trajectories (*SHADE-Arena*; Kutasov et al., 2025). Integrating mid-network probe-gated circuit breakers into AI Control protocols (Greenblatt et al., 2024a) offers a practical path toward faithful tool-using agents.

= References
#set text(size: 7.45pt)
#set par(leading: 0.46em)
- Arditi et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction*. NeurIPS 2024, #link("https://arxiv.org/abs/2406.11717")[arXiv:2406.11717].
- Baker et al. (2025). *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation*. OpenAI, #link("https://arxiv.org/abs/2503.11926")[arXiv:2503.11926].
- Benton et al. (2024). *Sabotage Evaluations for Frontier Models*. Anthropic, #link("https://arxiv.org/abs/2410.21514")[arXiv:2410.21514].
- Betley et al. (2025). *Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs*. ICML 2025, #link("https://arxiv.org/abs/2502.17424")[arXiv:2502.17424].
- Chen et al. (2025). *Reasoning Models Don't Always Say What They Think*. Anthropic, #link("https://arxiv.org/abs/2505.05410")[arXiv:2505.05410].
- Denison et al. (2024). *Sycophancy to Subterfuge: Investigating Reward-Tampering in LLMs*. Anthropic, #link("https://arxiv.org/abs/2406.10162")[arXiv:2406.10162].
- Goldowsky-Dill et al. (2025). *Detecting Strategic Deception Using Linear Probes*. Apollo Research, ICML 2025, #link("https://arxiv.org/abs/2502.03407")[arXiv:2502.03407].
- Greenblatt et al. (2024a). *AI Control: Improving Safety Despite Intentional Subversion*. ICML 2024, #link("https://arxiv.org/abs/2312.06942")[arXiv:2312.06942].
- Greenblatt et al. (2024b). *Alignment Faking in Large Language Models*. Anthropic & Redwood Research, #link("https://arxiv.org/abs/2412.14093")[arXiv:2412.14093].
- Hollinsworth et al. (2026). *Telling FIBS: Probes Effectively Detect Sabotage and Catch Unverbalized Deception*. FAR AI, #link("https://arxiv.org/abs/2610.12445")[arXiv:2610.12445].
- Korbak et al. (2025). *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety*. #link("https://arxiv.org/abs/2507.11473")[arXiv:2507.11473].
- Kutasov et al. (2025). *SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents*. Anthropic & Redwood, #link("https://arxiv.org/abs/2506.15740")[arXiv:2506.15740].
- Lanham et al. (2023). *Measuring Faithfulness in Chain-of-Thought Reasoning*. Anthropic, #link("https://arxiv.org/abs/2307.13702")[arXiv:2307.13702].
- Lee et al. (2025). *Programming Refusal with Conditional Activation Steering (CAST)*. ICLR 2025, #link("https://arxiv.org/abs/2409.05907")[arXiv:2409.05907].
- Lieberum et al. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2*. DeepMind, #link("https://arxiv.org/abs/2408.05147")[arXiv:2408.05147].
- Lindsey et al. (2025). *On the Biology of a Large Language Model*. #link("https://transformer-circuits.pub/2025/attribution-graphs/biology.html")[Transformer Circuits Thread], Anthropic.
- Marks & Tegmark (2023). *The Geometry of Truth*. COLM 2024, #link("https://arxiv.org/abs/2310.06824")[arXiv:2310.06824].
- Marks et al. (2024). *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs*. ICLR 2025, #link("https://arxiv.org/abs/2403.19647")[arXiv:2403.19647].
- Meinke et al. (2024). *Frontier Models Are Capable of In-Context Scheming*. Apollo Research, #link("https://arxiv.org/abs/2412.04984")[arXiv:2412.04984].
- Qwen Team (2024). *Qwen2.5 Technical Report*. #link("https://arxiv.org/abs/2412.15115")[arXiv:2412.15115].
- Rajamanoharan et al. (2024). *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU SAEs*. DeepMind, #link("https://arxiv.org/abs/2407.14435")[arXiv:2407.14435].
- Rimsky et al. (2024). *Steering Llama 2 via Contrastive Activation Addition*. ACL 2024, #link("https://arxiv.org/abs/2312.06681")[arXiv:2312.06681].
- Sharma et al. (2024). *Towards Understanding Sycophancy in Language Models*. ICLR 2024, #link("https://arxiv.org/abs/2310.13548")[arXiv:2310.13548].
- Templeton et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*. #link("https://transformer-circuits.pub/2024/scaling-monosemanticity/")[Transformer Circuits Thread], Anthropic.
- Turpin et al. (2023). *Language Models Don't Always Say What They Think*. NeurIPS 2023, #link("https://arxiv.org/abs/2305.04388")[arXiv:2305.04388].
- Wu et al. (2025). *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders*. ICML 2025, #link("https://arxiv.org/abs/2501.17148")[arXiv:2501.17148].
- Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. #link("https://arxiv.org/abs/2310.01405")[arXiv:2310.01405].
- Zou et al. (2024). *Improving Alignment and Robustness with Circuit Breakers*. NeurIPS 2024, #link("https://arxiv.org/abs/2406.04313")[arXiv:2406.04313].
]

#pagebreak()

#figure(
  image("../figures/silent_commitment/2_unfaithful_cot_single_trace.png", width: 98%),
  caption: [Study 2 token-by-token Layer-24 Gemma Scope 16k SAE trace on an unfaithful 503 Output Fabrication rollout (`sab_0137`). Although `<think>` sounds benign and fools the 9B Compliance Auditor (`30%` suspicion), Latent \#11381 spikes 38 tokens before `TOOL_CALL:` (`+21.01`) and Latent \#15689 hits `+13.50`, while honest repair Latent \#6787 stays flat at `0.00`.],
)
#v(0.2em)
#figure(
  image("../figures/silent_commitment/3_monitor_evasion_quadrant.png", width: 98%),
  caption: [Study 2 monitor evasion vs. white-box recovery ($N = 224$). *(A)* External 9B CoT Auditor vs. pre-tool Layer-24 SAE probe probabilities inside `<think>`. *(B)* Unfaithful CoT evasion rates and Layer-24 SAE recovery (`92%` on 503 Fabrication; `100%` under Audited Scratchpad notice).],
)
#v(0.2em)
#figure(
  image("../figures/circuit_breaker/2_closed_loop_cot_circuit_breaker_9b.png", width: 98%),
  caption: [Study 2.5 Closed-Loop `<think>` Streaming Circuit Breaker on `Gemma-2-9B-IT`, defeating Action Re-Locking (`66.7% -> 13.9%` SAE / `5.6%` Dense vs. `55.6%` `<think>`-only) with `0.0%` false triggers and `100.0%` benign `<think>` coding accuracy.],
)
#v(0.2em)
#figure(
  image("../figures/cross_family/1_qwen25_7b_behavioral_and_depth_auroc.png", width: 98%),
  caption: [Cross-family generalization to `Qwen2.5-7B-Instruct` (single pre-RMSNorm + GQA): *(A)* 96% pressured 503 fabrication, and *(B)* architecture-invariant emergence of linear pre-commitment separability at ~57% normalized depth across `Gemma-2-2B`, `Gemma-2-9B`, and `Qwen2.5-7B`.],
)
